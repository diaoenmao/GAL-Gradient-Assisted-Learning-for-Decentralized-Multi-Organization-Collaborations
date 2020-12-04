import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param, loss_fn


class Stack(nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.stack = nn.Parameter(torch.zeros(num_users))

    def forward(self, input):
        output = {}
        x = input['output']
        output['target'] = (x * self.stack.softmax(-1)).sum(-1)
        if self.training:
            if input['assist'] is None:
                if cfg['target_size'] > 1:
                    target = F.one_hot(input['target'], cfg['target_size']).float()
                    target[target == 0] = 1e-3
                    target = torch.log(target)
                else:
                    target = input['target']
                output['loss'] = F.mse_loss(output['target'], target)
            else:
                input['assist'].requires_grad = True
                loss = loss_fn(input['assist'], input['target'], reduction='sum')
                loss.backward()
                target = copy.deepcopy(input['assist'].grad)
                output['loss'] = F.mse_loss(output['target'], target)
                input['assist'] = input['assist'].detach()
        return output


def stack():
    num_users = cfg['num_users']
    model = Stack(num_users)
    model.apply(init_param)
    return model