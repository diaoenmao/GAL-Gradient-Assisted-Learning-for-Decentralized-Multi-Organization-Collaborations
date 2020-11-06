import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param, normalize, ce_loss, kd_loss


class Linear(nn.Module):
    def __init__(self, data_shape, classes_size):
        super().__init__()
        self.linear = nn.Linear(data_shape[0], classes_size)
        self.assist = nn.Linear(classes_size, classes_size)

    def forward(self, input):
        output = {}
        x = input['feature']
        x = normalize(x)
        if 'feature_split' in input:
            mask = torch.ones(x.size(1), device=x.device)
            mask[input['feature_split']] = 0
            x = torch.masked_fill(x, mask == 1, 0)
        output['score'] = self.linear(x)
        if 'assist' in input and cfg['assist'] == 'sum':
            assist = torch.zeros(input['assist'].size()[:2], device=input['assist'].device)
            for i in range(input['assist'].size(-1)):
                assist_i = input['assist'][:, :, i]
                valid = assist_i.sum(dim=-1) != 0
                assist[valid] += assist_i[valid]
            assist = assist / input['assist'].size(-1)
            output['score'] = 0.5 * output['score'] + 0.5 * self.assist(assist)
        output['loss'] = ce_loss(output['score'], input['label'])
        return output


def linear():
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    model = Linear(data_shape, classes_size)
    model.apply(init_param)
    return model