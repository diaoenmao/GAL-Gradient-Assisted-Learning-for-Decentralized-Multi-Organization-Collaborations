import copy
import torch
import torch.nn as nn
import numpy as np
from config import cfg
from .utils import init_param, normalize, loss_fn, feature_split, reset_parameters


class VAFL(nn.Module):
    def __init__(self, num_users, block, hidden_size, target_size, buffer_size):
        super().__init__()
        self.num_users = num_users
        blocks = []
        for i in range(num_users):
            block = copy.deepcopy(block)
            block.apply(reset_parameters)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.linear = nn.Linear(hidden_size * num_users, target_size)
        self.buffer_size = buffer_size
        buffer = {}
        for k in self.buffer_size:
            if cfg['data_name'] == 'MIMIC':
                buffer[k + '_'] = Buffer((num_users, self.buffer_size[k], 8, hidden_size))
            else:
                buffer[k + '_'] = Buffer((num_users, self.buffer_size[k], hidden_size))
        self.buffer = nn.ModuleDict(buffer)

    def detach(self):
        for k in self.buffer:
            self.buffer[k].detach()

    def forward(self, input):
        output = {}
        for i in range(self.num_users):
            if input['feature_split'][i] is not None:
                x_i = {'data': input['data'], 'feature_split': input['feature_split'][i]}
                x_i = self.blocks[i].feature(x_i)
                self.buffer[input['buffer'] + '_'].update(input['id'], i, x_i)
        x = self.buffer[input['buffer'] + '_'].get(input['id'], torch.arange(self.num_users, device=input['id'].device))
        if cfg['data_name'] == 'MIMIC':
            x = x.permute(1, 2, 0, 3)
            x = x.reshape(x.size(0), x.size(1), -1)
        else:
            x = x.permute(1, 0, 2)
            x = x.reshape(x.size(0), -1)
        output['target'] = self.linear(x)
        if cfg['data_name'] == 'MIMIC':
            output['target'] = output['target'].unsqueeze(0)
        if cfg['data_name'] == 'ModelNet40':
            input['target'] = input['target'].repeat(12 // cfg['num_users'])
        output['loss'] = loss_fn(output['target'], input['target'])
        return output


class Buffer(nn.Module):
    def __init__(self, buffer_size):
        super().__init__()
        self.register_buffer('buffer', torch.zeros(buffer_size))

    def update(self, sample_id, organization_id, input):
        self.buffer[organization_id, sample_id, :] = input
        return

    def get(self, sample_id, organization_id):
        return self.buffer[organization_id.view(organization_id.size(0), 1), sample_id.view(1, sample_id.size(0)), :]

    def detach(self):
        self.buffer.detach_()

    def forward(self):
        pass


def vafl(block, hidden_size):
    num_users = cfg['num_users']
    target_size = cfg['target_size']
    buffer_size = cfg['data_size']
    model = VAFL(num_users, block, hidden_size, target_size, buffer_size)
    model.apply(init_param)
    return model
