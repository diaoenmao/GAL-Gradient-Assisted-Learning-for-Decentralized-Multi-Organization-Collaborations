import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param, loss_fn


class Stack(nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.stack = nn.Linear(num_users, 1)

    def forward(self, input):
        output = {}
        x = input['output']
        output['target'] = self.stack(x).squeeze(-1)
        if self.training:
            if input['assist'] is None:
                output['loss'] = loss_fn(output['target'], input['target'])
            else:
                output['loss'] = loss_fn(input['assist'] - cfg['assist_rate'] * output['target'], input['target'])
        return output


def stack():
    num_users = cfg['num_users']
    model = Stack(num_users)
    model.apply(init_param)
    return model
