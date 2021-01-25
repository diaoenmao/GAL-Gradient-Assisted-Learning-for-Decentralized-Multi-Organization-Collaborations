import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg


def init_param(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    return m


def normalize(input):
    broadcast_size = [1] * input.dim()
    broadcast_size[1] = input.size(1)
    m, s = cfg['stats'][cfg['data_name']]
    m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
           torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
    input = input.sub(m).div(s)
    return input


def denormalize(input):
    broadcast_size = [1] * input.dim()
    broadcast_size[1] = input.size(1)
    m, s = cfg['stats'][cfg['data_name']]
    m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
           torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
    input = input.mul(s).add(m)
    return input


def feature_split(input, feature_split):
    if cfg['data_name'] in ['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR', 'MIMIC']:
        mask = torch.zeros(input.size(1), device=input.device)
        mask[feature_split] = 1
        output = torch.masked_fill(input, mask == 0, 0)
    elif cfg['data_name'] in ['MNIST', 'CIFAR10']:
        num_features = np.prod(cfg['data_shape']).item()
        mask = torch.zeros(num_features, device=input.device)
        mask[feature_split] = 1
        mask = mask.view(cfg['data_shape'])
        output = torch.masked_fill(input, mask == 0, 0)
    else:
        raise ValueError('Not valid data name')
    return output


def loss_fn(output, target, reduction='mean'):
    if cfg['target_size'] > 1:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        loss = F.mse_loss(output, target, reduction=reduction)
    return loss


def local_loss_fn(input, output, run):
    if run:
        if input['assist'] is None:
            output['loss_local'] = loss_fn(output['target'], input['target'])
            output['loss'] = loss_fn(output['target'], input['target'])
        else:
            input['assist'].requires_grad = True
            loss = loss_fn(input['assist'], input['target'], reduction='sum')
            loss.backward()
            target = - copy.deepcopy(input['assist'].grad)
            output['loss_local'] = F.mse_loss(output['target'], target)
            output['target'] = (input['assist'] + output['target']).detach()
            output['loss'] = loss_fn(output['target'], input['target'])
    else:
        output['target'] = input['assist']
        output['loss'] = loss_fn(output['target'], input['target'])
    return output


def assist_loss_fn(input, output, run):
    if run:
        if input['assist'] is None:
            output['loss'] = loss_fn(output['target'], input['target'])
        else:
            input['assist'].requires_grad = True
            loss = loss_fn(input['assist'], input['target'], reduction='sum')
            loss.backward()
            target = - copy.deepcopy(input['assist'].grad)
            output['loss'] = F.mse_loss(output['target'], target)
    return output
