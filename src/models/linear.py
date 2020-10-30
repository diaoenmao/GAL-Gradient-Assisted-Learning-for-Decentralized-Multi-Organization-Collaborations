import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param, normalize


class Linear(nn.Module):
    def __init__(self, data_shape, classes_size):
        super().__init__()
        self.linear = nn.Linear(data_shape[0], classes_size)

    def forward(self, input):
        output = {}
        x = input['feature']
        x = normalize(x)
        output['score'] = self.linear(x)
        output['loss'] = F.cross_entropy(output['score'], input['label'])
        return output


def linear():
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    model = Linear(data_shape, classes_size)
    model.apply(init_param)
    return model