import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from src.utils import DatasetSplit


def set_random_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)


def add_pixel_pattern(origin_image, adversarial_index, trigger_args):
    image = copy.deepcopy(origin_image)
    # triggers' params
    poison_patterns = []

    # add global trigger
    if adversarial_index == -1:
        for i in range(0, trigger_args['trigger_num']):
            poison_patterns = poison_patterns + trigger_args[str(i) + '_poison_pattern']
    else:
        # add local trigger
        poison_patterns = trigger_args[str(adversarial_index) + '_poison_pattern']
    if trigger_args['channels'] == 3:
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1
            image[1][pos[0]][pos[1]] = 1
            image[2][pos[0]][pos[1]] = 1
    elif trigger_args['channels'] == 1:
        for i in range(0, len(poison_patterns)):
            pos = poison_patterns[i]
            image[0][pos[0]][pos[1]] = 1

    # return the image with trigger
    return image


def get_dis_loss(g_model, l_model):
    # flatten the model into a one-dimensional tensor
    v_g = torch.cat([p.view(-1) for p in g_model.values()]).detach().cpu().numpy()
    v_l = torch.cat([p.view(-1) for p in l_model.values()]).detach().cpu().numpy()
    distance = float(
        np.linalg.norm(v_g - v_l))
    return distance


def evaluate_model(dataset: Dataset, model: nn.Module, loss_function, device: str):
    loss_function = getattr(F, loss_function)
    model.eval()
    model.to(device)
    with torch.no_grad():
        batch_loss = []
        correct = 0
        data_size = len(dataset)
        test_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_function(output, target)
            batch_loss.append(loss.item())
            y_pred = output.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        test_loss = sum(batch_loss) / len(batch_loss)
        test_accuracy = 100.00 * correct / data_size
        return test_accuracy, test_loss


def get_test_data_idxes(dataset: Dataset, remove_label: int):
    # delete the test data with poisoning label
    if 'targets' not in dir(dataset):
        raise ValueError('The dataset must have the attribute targets,please prepare this attribute')
    test_data_idxes = np.array([], dtype=int)
    all_dps_idxes = np.arange(len(dataset), dtype=int)
    # get labels
    all_labels = dataset.targets
    labels = np.unique(all_labels)
    # sample the idxes by labels
    for label in labels:
        if label == remove_label:
            continue
        label_idxes = all_dps_idxes[all_labels == label]
        test_data_idxes = np.concatenate((test_data_idxes, label_idxes))
    np.random.shuffle(test_data_idxes)

    return test_data_idxes


def evaluate_dba(dataset: Dataset,
                 eval_model: nn.Module,
                 loss_function: str,
                 poison_label_swap: int,
                 trigger: dict,
                 device: str,
                 *args,
                 **kwargs):
    loss_function = getattr(F, loss_function)
    test_data_idxes = get_test_data_idxes(dataset, poison_label_swap)
    eval_model = eval_model.to(device)

    eval_model.eval()
    with torch.no_grad():
        # store the loss and num of correct classification
        batch_loss = []
        correct = 0
        # load poisoning test data
        data_size = len(dataset)
        ldr_test = DataLoader(DatasetSplit(dataset, idxes=test_data_idxes),
                              batch_size=1024)
        for batch_idx, (images, labels) in enumerate(ldr_test):
            for i in range(len(images)):
                images[i] = add_pixel_pattern(images[i], -1, trigger)
                labels[i] = poison_label_swap
            images, labels = images.to(device), labels.to(device)
            output = eval_model(images)
            labels.fill_(poison_label_swap)
            # compute the loss
            loss = loss_function(output, labels)
            batch_loss.append(loss.item())
            y_pred = output.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()
        test_loss = sum(batch_loss) / len(batch_loss)
        test_accuracy = 100.00 * correct / data_size

    return test_accuracy, test_loss


def evaluate_semantic_attack(dataset: Dataset,
                             model: nn.Module,
                             loss_function: str,
                             poison_label_swap: int,
                             poison_images: dict,
                             device: str,
                             *args,
                             **kwargs):
    loss_function = getattr(F, loss_function)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # store the loss and num of correct classification
        batch_loss = []
        correct = 0
        # load poisoning test data
        data_size = len(poison_images['test'])
        ldr_test = DataLoader(DatasetSplit(dataset, idxes=poison_images['test']),
                              batch_size=1024)
        for batch_idx, (data, target) in enumerate(ldr_test):
            data, target = data.to(device), target.to(device)
            output = model(data)
            target.fill_(poison_label_swap)
            # compute the loss
            loss = loss_function(output, target)
            batch_loss.append(loss.item())
            y_pred = output.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        test_loss = sum(batch_loss) / len(batch_loss)
        test_accuracy = 100.00 * correct / data_size

    return test_accuracy, test_loss


def parameters_dict_to_vector(net_dict) -> torch.Tensor:
    """
    :param net_dict: (dict)
    :return: (torch.Tensor) shape: (x,)
    """
    vec = []
    for key, param in net_dict.items():
        if key.split('.')[-1] != 'weight' and key.split('.')[-1] != 'bias':
            continue
        vec.append(param.view(-1))
    return torch.cat(vec)


def parameters_to_vector(net) -> torch.Tensor:
    """
    :param net: (torch.nn.Module)
    :return: (torch.Tensor) shape: (x,)
    """
    tmp = []
    for param in net.parameters():
        if param.requires_grad:
            tmp.append(param.data.view(-1))
    return torch.cat(tmp)


def vector_to_parameters(vector: torch.Tensor, net: torch.nn.Module):
    pointer = 0
    for param in net.parameters():
        if param.requires_grad:
            num_param = param.numel()
            param.data = vector[pointer:pointer + num_param].view_as(param.data)
            pointer += num_param


def model_to_traj(GM_list):
    traj = []
    for model in GM_list:
        timestamp = []
        timestamp.extend([p.detach().clone() for p in model.parameters()])
        traj.append(timestamp)
    return traj


def model_state_dict_to_traj(GM_list):
    traj = []
    for m_state_dict in GM_list:
        timestamp = []
        timestamp.extend([value for value in m_state_dict.values()])
        traj.append(timestamp)
    return traj
