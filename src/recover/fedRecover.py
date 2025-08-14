import copy
import time

import torch

from .recoverBase import RecoverBase
from ..aggregator import average_weights
from ..aggregator.average import average_weights_by_state_dict
from ..utils import setup_logger
from ..utils.helper import parameters_to_vector, parameters_dict_to_vector, evaluate_model

logger = setup_logger()


def lbfgs(S_k_list, Y_k_list, v):
    """
    L-BFGS helper function to compute the approximate inverse Hessian-vector product (H_k * v)
    """
    # Concatenate historical vectors into matrices
    curr_S_k = torch.cat(S_k_list, dim=1)  # [dim, history_size]
    curr_Y_k = torch.cat(Y_k_list, dim=1)

    # Compute intermediate matrices
    S_k_time_Y_k = torch.matmul(curr_S_k.t(), curr_Y_k)  # [history_size, history_size]
    S_k_time_S_k = torch.matmul(curr_S_k.t(), curr_S_k)

    # Decompose into upper and lower triangular parts
    R_k = torch.triu(S_k_time_Y_k.cpu()).numpy()  # Extract upper triangular to CPU
    L_k = S_k_time_Y_k - torch.tensor(R_k, device='cuda:0')  # Reconstruct lower triangular

    # Compute scaling factor sigma_k (initial Hessian approximation)
    sigma_k = torch.matmul(Y_k_list[-1].t(), S_k_list[-1]) / torch.matmul(S_k_list[-1].t(), S_k_list[-1])

    # Construct block matrix components
    D_k_diag = torch.diag(S_k_time_Y_k)
    upper_mat = torch.cat([sigma_k * S_k_time_S_k, L_k], dim=1)
    lower_mat = torch.cat([L_k.t(), -torch.diag(D_k_diag)], dim=1)
    mat = torch.cat([upper_mat, lower_mat], dim=0)

    # Compute matrix inverse (with potential regularization for stability)
    mat_inv = torch.inverse(mat)

    # Compute the approximate product using two-loop recursion
    approx_prod = sigma_k * v  # Initial Hessian approximation

    # Construct right-hand side vector for linear system
    p_mat = torch.cat([
        torch.matmul(curr_S_k.t(), sigma_k * v),
        torch.matmul(curr_Y_k.t(), v)
    ], dim=0)

    # Apply inverse matrix and subtract from initial approximation
    approx_prod -= torch.matmul(
        torch.matmul(
            torch.cat([sigma_k * curr_S_k, curr_Y_k], dim=1),
            mat_inv
        ),
        p_mat
    )

    return approx_prod


class FedRecover(RecoverBase):
    def __init__(self,
                 test_dataset,
                 global_model,
                 clients_pool,
                 old_global_models,
                 old_client_models,
                 select_info,
                 malicious_clients,
                 recover_config,
                 loss_function,
                 attack_config,
                 *args,
                 **kwargs):
        super().__init__(
            test_dataset,
            global_model,
            clients_pool,
            old_global_models,
            old_client_models,
            select_info,
            malicious_clients,
            loss_function,
            attack_config)
        self.rounds = recover_config['rounds']
        self.T_w = recover_config['warm_up_rounds']
        self.T_f = recover_config['final_tuning_rounds']
        self.T_mid = self.rounds - self.T_w - self.T_f
        self.T_c = recover_config['correction_period']
        self.buffer_size = recover_config['buffer_size']
        self.prev_train_loss = 10
        self.w_diff_buffer = []
        self.g_diff_buffer = dict()
        self.round_losses = []
        self.MA = []
        self.BA = []

    def recover(self):

        # warm up stage
        self.warm_up()
        # middle stage
        self.mid_stage()
        # final tuning stage
        self.final_tuning()

        logger.info("----- The recover process end -----")
        logger.info(f"Total time cost: {self.time_cost}s")
        logger.debug(f'Main Accuracy:{self.MA}')
        logger.debug(f'Backdoor Accuracy:{self.BA}')

        return self.global_model

    def warm_up(self):
        # get the initial global model
        self.global_model.load_state_dict(self.old_global_models[0])
        for rd in range(self.T_w):
            start_time = time.time()
            # select remaining clients
            remaining_clients_id, remaining_clients_models = self.remove_malicious_clients(
                self.select_info[rd],
                self.old_client_models[rd])
            # begin training
            logger.info("----- FedRecover Warm up Round {:3d}  -----".format(rd))
            logger.info(f'remaining client:{remaining_clients_id}')
            # store the local loss and local model for each client
            local_losses = []
            local_models = []
            # distribution and local training
            for client_id in remaining_clients_id:
                client = self.clients_pool[client_id]
                client.receive_global_model(self.global_model)
                local_model, local_loss = client.local_train()
                local_models.append(local_model)
                local_losses.append(local_loss)

            # update the buffer that stores the difference of  client grad
            # flatten the model into a one-dimensional tensor
            v_global_model_new_last = parameters_to_vector(self.global_model)
            v_global_model_old = parameters_dict_to_vector(self.old_global_models[rd])
            for i, local_model in enumerate(local_models):
                V_client_model_new = parameters_to_vector(local_models[i])
                V_client_model_old = parameters_dict_to_vector(remaining_clients_models[i])
                g_diff = (V_client_model_new - v_global_model_new_last) - \
                         (V_client_model_old - v_global_model_old)
                if remaining_clients_id[i] not in self.g_diff_buffer.keys():
                    self.g_diff_buffer[remaining_clients_id[i]] = [g_diff.unsqueeze(1)]
                else:
                    if len(self.g_diff_buffer[remaining_clients_id[i]]) == self.buffer_size:
                        self.g_diff_buffer[remaining_clients_id[i]].pop(0)
                    self.g_diff_buffer[remaining_clients_id[i]].append(g_diff.unsqueeze(1))

            # aggregation
            self.global_model = average_weights(self.global_model, local_models)

            # update the buffer that stores the difference of global model
            # flatten the model into a one-dimensional tensor
            v_global_model_new = parameters_to_vector(self.global_model)
            w_diff = v_global_model_new - v_global_model_old
            if len(self.w_diff_buffer) == self.buffer_size:
                self.w_diff_buffer.pop(0)
            self.w_diff_buffer.append(w_diff.unsqueeze(1))

            # compute the average loss in a round
            round_loss = sum(local_losses) / len(local_losses)
            logger.info('Training average loss: {:.3f}'.format(round_loss))
            self.round_losses.append(round_loss)

            self.time_cost += time.time() - start_time

            # evaluate the global model
            test_accuracy, test_loss = evaluate_model(
                dataset=self.test_dataset,
                model=self.global_model,
                loss_function=self.loss_function,
                device='cuda'
            )
            logger.info("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))
            self.MA.append(round(test_accuracy.item(), 2))
            attack_accuracy = self.eval_attack()
            logger.info("Attack accuracy: {:.2f}%".format(attack_accuracy))
            self.BA.append(round(attack_accuracy.item(), 2))

    def mid_stage(self):
        correction_round = [i for i in range(self.T_w - 1, self.rounds - self.T_f, self.T_c)]
        correction_round.pop(0)
        for rd in range(self.T_w, self.rounds - self.T_f):
            start_time = time.time()
            # select remaining clients
            remaining_clients_id, remaining_clients_models, = self.remove_malicious_clients(
                self.select_info[rd],
                self.old_client_models[rd])
            # begin training
            logger.info("----- FedRecover Middle Round {:3d}  -----".format(rd))
            logger.info(f'remaining client:{remaining_clients_id}')
            local_losses = []
            local_models = []

            if rd in correction_round:
                # correction
                # distribution and local training
                for client_id in remaining_clients_id:
                    client = self.clients_pool[client_id]
                    client.receive_global_model(self.global_model)
                    local_model, local_loss = client.local_train()
                    local_models.append(local_model)
                    local_losses.append(local_loss)
                # update the buffer that stores the difference of  client grad
                # flatten the model into a one-dimensional tensor
                v_global_model_new_last = parameters_to_vector(self.global_model)
                v_global_model_old = parameters_dict_to_vector(self.old_global_models[rd])
                for i, local_model in enumerate(local_models):
                    V_client_model_new = parameters_to_vector(local_models[i])
                    V_client_model_old = parameters_dict_to_vector(remaining_clients_models[i])
                    g_diff = (V_client_model_new - v_global_model_new_last) - \
                             (V_client_model_old - v_global_model_old)
                    if remaining_clients_id[i] not in self.g_diff_buffer.keys():
                        self.g_diff_buffer[remaining_clients_id[i]] = [g_diff.unsqueeze(1)]
                    else:
                        if len(self.g_diff_buffer[remaining_clients_id[i]]) == self.buffer_size:
                            self.g_diff_buffer[remaining_clients_id[i]].pop(0)
                        self.g_diff_buffer[remaining_clients_id[i]].append(g_diff.unsqueeze(1))
                # aggregation
                self.global_model = average_weights(self.global_model, local_models)

                # update the buffer that stores the difference of global model
                # flatten the model into a one-dimensional tensor
                v_global_model_new = parameters_to_vector(self.global_model)
                w_diff = v_global_model_new - v_global_model_old
                if len(self.w_diff_buffer) == self.buffer_size:
                    self.w_diff_buffer.pop(0)
                self.w_diff_buffer.append(w_diff.unsqueeze(1))
            else:
                # predict the grad for each client in this round
                hpvs = []
                v_new_global_model = parameters_to_vector(self.global_model).unsqueeze(1)
                v_old_global_model = parameters_dict_to_vector(self.old_global_models[rd]).unsqueeze(1)
                w_diff = v_new_global_model - v_old_global_model
                for i in range(len(remaining_clients_id)):
                    if remaining_clients_id[i] not in self.g_diff_buffer.keys():
                        hpvs.append(None)
                        continue
                    hpv = lbfgs(self.w_diff_buffer, self.g_diff_buffer[remaining_clients_id[i]], w_diff)
                    hpvs.append(hpv)
                for i, old_CM in enumerate(remaining_clients_models):
                    if hpvs[i] is None:
                        local_models.append(remaining_clients_models[i])
                        local_losses.append(0)
                        continue
                    v_old_CM = parameters_dict_to_vector(old_CM).unsqueeze(1)
                    v_pred_CM = v_new_global_model + 0.5 * (v_old_CM - v_old_global_model + hpvs[i])
                    pred_CM = copy.deepcopy(self.global_model)
                    pointer = 0
                    for param in pred_CM.parameters():
                        num_param = param.numel()
                        param.data = v_pred_CM[pointer:pointer + num_param].view_as(param.data)
                        pointer += num_param
                    local_models.append(pred_CM.state_dict())
                    local_losses.append(0)

                # aggregation
                self.global_model.load_state_dict(average_weights_by_state_dict(local_models))

            # compute the average loss in a round
            round_loss = sum(local_losses) / len(local_losses)
            logger.info('Training average loss: {:.3f}'.format(round_loss))
            self.round_losses.append(round_loss)

            self.time_cost += time.time() - start_time

            # evaluate the global model
            test_accuracy, test_loss = evaluate_model(
                dataset=self.test_dataset,
                model=self.global_model,
                loss_function=self.loss_function,
                device='cuda'
            )
            logger.info("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))
            self.MA.append(round(test_accuracy.item(), 2))
            attack_accuracy = self.eval_attack()
            logger.info("Attack accuracy: {:.2f}%".format(attack_accuracy))
            self.BA.append(round(attack_accuracy.item(), 2))

    def final_tuning(self):
        # get the initial global model
        self.global_model.load_state_dict(self.old_global_models[0])
        for rd in range(self.rounds - self.T_f, self.rounds):
            start_time = time.time()
            # select remaining clients
            remaining_clients_id, remaining_clients_models = self.remove_malicious_clients(
                self.select_info[rd],
                self.old_client_models[rd])
            # begin training
            logger.info("----- FedRecover Final tuning Round {:3d}  -----".format(rd))
            logger.info(f'remaining client:{remaining_clients_id}')
            # store the local loss and local model for each client
            local_losses = []
            local_models = []
            # distribution and local training
            for client_id in remaining_clients_id:
                client = self.clients_pool[client_id]
                client.receive_global_model(self.global_model)
                local_model, local_loss = client.local_train()
                local_models.append(local_model)
                local_losses.append(local_loss)

            # aggregation
            self.global_model = average_weights(self.global_model, local_models)

            # compute the average loss in a round
            round_loss = sum(local_losses) / len(local_losses)
            logger.info('Training average loss: {:.3f}'.format(round_loss))
            self.round_losses.append(round_loss)

            self.time_cost += time.time() - start_time

            # evaluate the global model
            test_accuracy, test_loss = evaluate_model(
                dataset=self.test_dataset,
                model=self.global_model,
                loss_function=self.loss_function,
                device='cuda'
            )
            logger.info("Testing accuracy: {:.2f}%, loss: {:.3f}".format(test_accuracy, test_loss))
            self.MA.append(round(test_accuracy.item(), 2))
            attack_accuracy = self.eval_attack()
            logger.info("Attack accuracy: {:.2f}%".format(attack_accuracy))
            self.BA.append(round(attack_accuracy.item(), 2))

