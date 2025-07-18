import copy

import torch


@torch.no_grad()
def average_weights(global_model, local_models):
    global_params = list(global_model.parameters())
    global_param_sums = [torch.zeros_like(param) for param in global_params]

    for model in local_models:
        for global_sum, local_param in zip(global_param_sums, model.parameters()):
            global_sum.add_(local_param)

    for global_param, param_sum in zip(global_params, global_param_sums):
        global_param.copy_(param_sum / len(local_models))

    return global_model


def average_weights_by_state_dict(local_models_state_dict):
    global_model_state_dict = {}
    average_state_dict = copy.deepcopy(local_models_state_dict[0])
    for k in average_state_dict.keys():
        if k.split('.')[-1] != 'weight' and k.split('.')[-1] != 'bias':
            k_ = k.split("_module.", 1)[-1]
            global_model_state_dict[k_] = average_state_dict[k]
            continue
        for i in range(1, len(local_models_state_dict)):
            average_state_dict[k] += local_models_state_dict[i][k]
        average_state_dict[k] /= len(local_models_state_dict)
        k_ = k.split("_module.", 1)[-1]
        global_model_state_dict[k_] = average_state_dict[k]
    return global_model_state_dict
