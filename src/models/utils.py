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
    if cfg['data_name'] in cfg['stats']:
        broadcast_size = [1] * input.dim()
        broadcast_size[1] = input.size(1)
        m, s = cfg['stats'][cfg['data_name']]
        m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
               torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
        input = input.sub(m).div(s)
    return input


def denormalize(input):
    if cfg['data_name'] in cfg['stats']:
        broadcast_size = [1] * input.dim()
        broadcast_size[1] = input.size(1)
        m, s = cfg['stats'][cfg['data_name']]
        m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
               torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
        input = input.mul(s).add(m)
    return input


def feature_split(input, feature_split):
    if cfg['data_name'] in ['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR', 'MIMIC']:
        mask = torch.zeros(input.size(-1), device=input.device)
        mask[feature_split] = 1
        output = torch.masked_fill(input, mask == 0, 0)
    elif cfg['data_name'] in ['MNIST', 'CIFAR10']:
        num_features = np.prod(cfg['data_shape']).item()
        mask = torch.zeros(num_features, device=input.device)
        mask[feature_split] = 1
        mask = mask.view(cfg['data_shape'])
        output = torch.masked_fill(input, mask == 0, 0)
    elif cfg['data_name'] in ['ModelNet40']:
        output = torch.index_select(input, -1, feature_split)
        output = output.permute(4, 0, 1, 2, 3).reshape(-1, *output.size()[1:-1])
    else:
        raise ValueError('Not valid data name')
    return output


def loss_fn(output, target, reduction='mean', loss_mode=None):
    if target.dtype == torch.int64:
        loss = F.cross_entropy(output, target, reduction=reduction)
    else:
        if loss_mode is None:
            if cfg['data_name'] in ['Diabetes', 'BostonHousing', 'MIMIC']:
                loss = F.l1_loss(output, target, reduction=reduction)
            else:
                loss = F.mse_loss(output, target, reduction=reduction)
        else:
            if loss_mode == 'l1':
                loss = F.l1_loss(output, target, reduction=reduction)
            elif loss_mode == 'l1.5':
                if reduction == 'sum':
                    loss = (x - y).abs().pow(1.5).sum()
                else:
                    loss = (x - y).abs().pow(1.5).mean()
            elif loss_mode == 'l2':
                loss = F.mse_loss(output, target, reduction=reduction)
            elif loss_mode == 'l4':
                if reduction == 'sum':
                    loss = (x - y).abs().pow(4).sum()
                else:
                    loss = (x - y).abs().pow(4).mean()
            else:
                raise ValueError('Not valid loss mode')
    return loss


def reset_parameters(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()
