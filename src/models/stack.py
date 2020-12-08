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
        # self.stack = nn.Linear(num_users, 1)
        # self.assist_rate = nn.Parameter(torch.ones(1) * cfg['assist_rate'])

    def forward(self, input):
        output = {}
        x = input['output']
        output['target'] = (x * self.stack.softmax(-1)).sum(-1)
        # output['target'] = self.stack(x).squeeze(-1)
        if self.training:
            if input['assist'] is None:
                # if cfg['target_size'] > 1:
                #     target = F.one_hot(input['target'], cfg['target_size']).float()
                #     target[target == 0] = 1e-3
                #     target = torch.log(target)
                # else:
                #     target = input['target']
                # output['loss'] = F.mse_loss(output['target'], target)
                output['loss'] = loss_fn(output['target'], input['target'])
            else:
                # input['assist'].requires_grad = True
                # loss = loss_fn(input['assist'], input['target'], reduction='sum')
                # loss.backward()
                # target = copy.deepcopy(input['assist'].grad)
                # output['loss'] = F.mse_loss(output['target'], target)

                # if cfg['target_size'] > 1:
                #     target = F.one_hot(input['target'], cfg['target_size']).float()
                #     target[target == 0] = 1e-3
                #     target = torch.log(target)
                # else:
                #     target = input['target']
                # input['assist'] = input['assist'].detach()
                # output['loss'] = F.mse_loss(input['assist'] - cfg['assist_rate'] * output['target'], target)
                output['loss'] = loss_fn(input['assist'] - 1 * output['target'], input['target'])
                # output['loss'] = loss_fn(input['assist'] - self.assist_rate * output['target'], input['target'])
                output['loss'] = loss_fn(output['target'], input['target'])
        return output


def stack():
    num_users = cfg['num_users']
    model = Stack(num_users)
    model.apply(init_param)
    return model
