import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param


class Stack(nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.stack = nn.Linear(num_users, 1)

    def forward(self, input):
        output = {}
        x = input['score']
        output['score'] = self.stack(x).squeeze(-1)
        if self.training:
            if input['assist'] is None:
                target = F.one_hot(input['target'], cfg['classes_size']).float()
                target[target == 0] = 1e-4
                target = torch.log(target)
                output['loss'] = F.mse_loss(output['score'], target)
            else:
                input['assist'].requires_grad = True
                loss = F.cross_entropy(input['assist'], input['target'], reduction='sum')
                loss.backward()
                target = copy.deepcopy(input['assist'].grad)
                output['loss'] = F.mse_loss(output['score'], target)
                input['assist'] = input['assist'].detach()
        return output


def stack():
    num_users = cfg['num_users']
    model = Stack(num_users)
    model.apply(init_param)
    return model