import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import cfg
from .utils import init_param, normalize, loss_fn, feature_split


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


def linear():
    data_shape = cfg['data_shape']
    target_size = cfg['target_size']
    model = Linear(data_shape, target_size)
    model.apply(init_param)
    return model