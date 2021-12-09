import copy
import torch
import torch.nn as nn
import numpy as np
from config import cfg
from .utils import init_param, normalize, loss_fn, feature_split, reset_parameters


class DL(nn.Module):
    def __init__(self, block, global_epoch, hidden_size, target_size):
        super().__init__()
        self.target_size = target_size
        self.block = copy.deepcopy(block)
        self.block.apply(reset_parameters)
        linear = []
        for i in range(global_epoch):
            linear.append(nn.Linear(hidden_size, target_size))
        self.linear = nn.ModuleList(linear)

    def forward(self, input):
        output = {'loss': 0}
        x = {'data': input['data'], 'feature_split': input['feature_split']}
        x = self.block.feature(x)
        num_epochs = input['target'].size(1)
        output_target = []
        input_target = []
        for i in range(num_epochs):
            output_target_i = self.linear[i](x)
            if cfg['data_name'] == 'MIMIC':
                output_target_i = output_target_i.unsqueeze(0)
            input_target_i = input['target'][:, i]
            if cfg['data_name'] == 'ModelNet40':
                input_target_i = input_target_i.repeat(12 // cfg['num_users'], 1)
            if 'loss_mode' in input:
                output['loss'] += loss_fn(output_target_i, input_target_i, loss_mode=input['loss_mode'])
            else:
                output['loss'] += loss_fn(output_target_i, input_target_i)
            output_target.append(output_target_i)
            input_target.append(input_target_i)
        output['target'] = torch.stack(output_target, dim=1)
        input['target'] = torch.stack(input_target, dim=1)
        return output


def dl(block, hidden_size):
    global_epoch = cfg['global_epoch']
    target_size = cfg['target_size']
    model = DL(block, global_epoch, hidden_size, target_size)
    model.apply(init_param)
    return model
