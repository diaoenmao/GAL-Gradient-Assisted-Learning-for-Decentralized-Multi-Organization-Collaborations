import torch
import torch.nn as nn
import numpy as np
from config import cfg
from .utils import init_param, normalize, loss_fn, local_loss_fn, feature_split


class Linear(nn.Module):
    def __init__(self, data_shape, target_size):
        super().__init__()
        self.linear = nn.Linear(np.prod(data_shape).item(), target_size)

    def forward(self, input):
        output = {}
        x = input['data']
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        x = x.view(x.size(0), -1)
        output['target'] = self.linear(x)
        if 'assist' in input:
            output = local_loss_fn(input, output, self.training)
        else:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def linear():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    model = Linear(data_shape, target_size)
    model.apply(init_param)
    return model
