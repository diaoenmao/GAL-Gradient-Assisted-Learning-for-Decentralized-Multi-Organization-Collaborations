import copy
import torch
import torch.nn as nn
import numpy as np
from config import cfg
from .utils import init_param, normalize, loss_fn, feature_split, reset_parameters, unpad_sequence
from torch.nn.utils.rnn import pad_sequence


class VAFL(nn.Module):
    def __init__(self, num_users, block, hidden_size, target_size, num_samples):
        super().__init__()
        self.num_users = num_users
        blocks = []
        for i in range(num_users):
            block = copy.deepcopy(block)
            block.apply(reset_parameters)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.linear = nn.Linear(hidden_size * num_users, target_size)
        self.num_samples = num_samples
        buffer = {}
        for k in self.num_samples:
            buffer[k + '_'] = Buffer(k, num_users, self.num_samples[k], hidden_size)
        self.buffer = nn.ModuleDict(buffer)

    def detach(self):
        for k in self.buffer:
            self.buffer[k].detach()

    def forward(self, input):
        output = {}
        for i in range(self.num_users):
            if input['feature_split'][i] is not None:
                if cfg['data_name'] in ['MIMICL', 'MIMICM']:
                    x_i = {'data': input['data'], 'length': input['length'], 'feature_split': input['feature_split'][i]}
                    x_i = self.blocks[i].feature(x_i)
                    x_i = unpad_sequence(x_i, input['length'])
                    self.buffer[input['buffer'] + '_'].update(i, input['id'], x_i)
                else:
                    x_i = {'data': input['data'], 'feature_split': input['feature_split'][i]}
                    x_i = self.blocks[i].feature(x_i)
                    self.buffer[input['buffer'] + '_'].update(i, input['id'], x_i)
        x = self.buffer[input['buffer'] + '_'].get(torch.arange(self.num_users, device=input['id'].device), input['id'])
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            x = x.permute(1, 2, 0, 3)
            x = x.reshape(x.size(0), x.size(1), -1)
        else:
            x = x.permute(1, 0, 2)
            x = x.reshape(x.size(0), -1)
        output['target'] = self.linear(x)
        if cfg['data_name'] in ['ModelNet40', 'ShapeNet55']:
            input['target'] = input['target'].repeat(12 // cfg['num_users'])
        output['loss'] = loss_fn(output['target'], input['target'])
        return output


class Buffer(nn.Module):
    def __init__(self, split, num_users, num_samples, hidden_size):
        super().__init__()
        self.split = split
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            self.buffer = [[torch.zeros((cfg['data_length'][split][i], hidden_size)) for i in
                            range(num_samples)] for _ in range(num_users)]
        else:
            self.register_buffer('buffer', torch.zeros(num_users, num_samples, hidden_size))

    def update(self, organization_id, sample_id, input):
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            for i in range(len(sample_id)):
                self.buffer[organization_id][sample_id[i]] = input[i].to('cpu')
        else:
            self.buffer[organization_id, sample_id, :] = input
        return

    def get(self, organization_id, sample_id):
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            buffer = []
            for i in range(len(organization_id)):
                buffer_i = []
                for j in range(len(sample_id)):
                    buffer_i_j = self.buffer[organization_id[i]][sample_id[j]]
                    buffer_i.append(buffer_i_j)
                buffer_i = pad_sequence(buffer_i, batch_first=True, padding_value=0)
                buffer.append(buffer_i)
            buffer = torch.stack(buffer, dim=0).to(cfg['device'])
        else:
            buffer = self.buffer[organization_id.view(organization_id.size(0), 1),
                     sample_id.view(1, sample_id.size(0)), :]
        return buffer

    def detach(self):
        if cfg['data_name'] in ['MIMICL', 'MIMICM']:
            for i in range(len(self.buffer)):
                for j in range(len(self.buffer[i])):
                    self.buffer[i][j] = self.buffer[i][j].detach()
        else:
            self.buffer.detach_()

    def forward(self):
        pass


def vafl(block, hidden_size):
    num_users = cfg['num_users']
    target_size = cfg['target_size']
    num_samples = cfg['data_size']
    model = VAFL(num_users, block, hidden_size, target_size, num_samples)
    model.apply(init_param)
    return model
