import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import init_param, normalize, loss_fn, feature_split
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
            if self.training:
                if input['assist'] is None:
                    target = F.one_hot(input['target'], cfg['target_size']).float()
                    target[target == 0] = 1e-4
                    target = torch.log(target)
                    output['loss_local'] = F.mse_loss(output['target'], target)
                    output['loss'] = loss_fn(output['target'], input['target'])
                else:
                    input['assist'].requires_grad = True
                    loss = loss_fn(input['assist'], input['target'], reduction='sum')
                    loss.backward()
                    target = copy.deepcopy(input['assist'].grad)
                    output['loss_local'] = F.mse_loss(output['target'], target)
                    input['assist'] = input['assist'].detach()
                    output['target'] = input['assist'] - cfg['assist_rate'] * output['target']
                    output['loss'] = loss_fn(output['target'], input['target'])
            else:
                output['target'] = input['assist']
                output['loss'] = loss_fn(output['target'], input['target'])
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