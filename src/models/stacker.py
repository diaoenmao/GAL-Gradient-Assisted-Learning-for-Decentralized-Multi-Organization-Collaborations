import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param, normalize, ce_loss, kd_loss


class Stacker(nn.Module):
    def __init__(self, num_users):
        super().__init__()
        self.stacker = nn.Linear(num_users, 1)

    def forward(self, input):
        output = {}
        x = input['scores']
        output['score'] = self.stacker(x).squeeze(-1)
        output['score'] = input['assist']
        output['loss'] = F.cross_entropy(output['score'], input['label'])
        return output


def stacker():
    num_users = cfg['num_users']
    model = Stacker(num_users)
    model.apply(init_param)
    return model