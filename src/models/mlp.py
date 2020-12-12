import torch
import torch.nn as nn
import numpy as np
from .utils import init_param, normalize, loss_fn, local_loss_fn, feature_split
from config import cfg


class MLPBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.linear(self.activation(self.bn(input)))
        return output


class MLP(nn.Module):
    def __init__(self, data_shape, hidden_size, target_size):
        super().__init__()
        blocks = [nn.Linear(np.prod(data_shape).item(), hidden_size[0])]
        for i in range(len(hidden_size) - 1):
            blocks.append(MLPBlock(hidden_size[i], hidden_size[i + 1]))
        blocks.extend([
            nn.BatchNorm1d(hidden_size[-1]),
            nn.ReLU(),
            nn.Linear(hidden_size[-1], target_size),
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = {}
        x = input['data']
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        x = x.view(x.size(0), -1)
        output['target'] = self.blocks(x)
        if 'assist' in input:
            output = local_loss_fn(input, output, self.training)
        else:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def mlp():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    hidden_size = cfg['mlp']['hidden_size']
    model = MLP(data_shape, hidden_size, target_size)
    model.apply(init_param)
    return model
