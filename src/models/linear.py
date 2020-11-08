import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param, normalize, ce_loss, kd_loss


class Linear(nn.Module):
    def __init__(self, data_shape, classes_size):
        super().__init__()
        self.linear = nn.Linear(data_shape[0], classes_size)

    def forward(self, input):
        output = {}
        x = input['feature']
        x = normalize(x)
        if 'feature_split' in input:
            mask = torch.ones(x.size(1), device=x.device)
            mask[input['feature_split']] = 0
            x = torch.masked_fill(x, mask == 1, 0)
        output['score'] = self.linear(x)
        if 'assist' in input:
            if self.training:
                if input['assist'] is None:
                    target = F.one_hot(input['label'], cfg['classes_size']).float()
                    target[target == 0] = 1e-10
                    target = torch.log(target)
                    output['loss'] = F.mse_loss(output['score'], target)
                else:
                    input['assist'].requires_grad = True
                    loss = F.cross_entropy(input['assist'], input['label'], reduction='sum')
                    loss.backward()
                    target = copy.deepcopy(input['assist'].grad)
                    output['loss'] = F.mse_loss(output['score'], target)
                    input['assist'] = input['assist'].detach()
                    output['score'] = input['assist'] - cfg['assist_rate'] * output['score']
            else:
                output['score'] = input['assist']
                output['loss'] = F.cross_entropy(output['score'], input['label'])
        else:
            output['loss'] = F.cross_entropy(output['score'], input['label'])
        return output


def linear():
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    model = Linear(data_shape, classes_size)
    model.apply(init_param)
    return model