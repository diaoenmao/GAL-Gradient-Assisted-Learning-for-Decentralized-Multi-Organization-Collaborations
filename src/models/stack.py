import torch
import torch.nn as nn
from config import cfg
from .utils import assist_loss_fn


class Stack(nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.stack = nn.Parameter(torch.zeros(num_users))

    def forward(self, input):
        output = {}
        x = input['output']
        output['target'] = (x * self.stack.softmax(-1)).sum(-1)
        output = assist_loss_fn(input, output, self.training)
        return output


def stack():
    num_users = cfg['num_users']
    model = Stack(num_users)
    return model
