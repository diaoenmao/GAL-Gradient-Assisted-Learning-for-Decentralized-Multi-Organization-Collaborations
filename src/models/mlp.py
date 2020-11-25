import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import init_param, normalize, ce_loss, kd_loss
from config import cfg


class MLPBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.linear(self.activation(self.bn(input)))
        return output


class MLP(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size):
        super().__init__()
        if len(hidden_size) == 0:
            self.blocks = nn.Linear(data_shape[0], classes_size)
        else:
            blocks = [nn.Linear(data_shape[0], hidden_size[0])]
            for i in range(len(hidden_size) - 1):
                blocks.append(MLPBlock(hidden_size[i], hidden_size[i + 1]))
            blocks.extend([
                nn.BatchNorm1d(hidden_size[-1]),
                nn.ReLU(),
                nn.Linear(hidden_size[-1], classes_size),
            ])
            self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = {}
        x = input['feature']
        x = normalize(x)
        if 'feature_split' in input:
            mask = torch.ones(x.size(1), device=x.device)
            mask[input['feature_split']] = 0
            x = torch.masked_fill(x, mask == 1, 0)
        output['score'] = self.blocks(x)
        if 'assist' in input:
            if self.training:
                if input['assist'] is None:
                    target = F.one_hot(input['label'], cfg['classes_size']).float()
                    target[target == 0] = 1e-4
                    target = torch.log(target)
                    output['loss_local'] = F.mse_loss(output['score'], target)
                    output['loss'] = F.cross_entropy(output['score'], input['label'])
                else:
                    input['assist'].requires_grad = True
                    loss = F.cross_entropy(input['assist'], input['label'], reduction='sum')
                    loss.backward()
                    target = copy.deepcopy(input['assist'].grad)
                    output['loss_local'] = F.mse_loss(output['score'], target)
                    input['assist'] = input['assist'].detach()
                    output['score'] = input['assist'] - cfg['assist_rate'] * output['score']
                    output['loss'] = F.cross_entropy(output['score'], input['label'])
            else:
                output['score'] = input['assist']
                output['loss'] = F.cross_entropy(output['score'], input['label'])
        else:
            output['loss'] = F.cross_entropy(output['score'], input['label'])
        return output


def mlp():
    data_shape = cfg['data_shape']
    classes_size = cfg['classes_size']
    hidden_size = cfg['mlp']['hidden_size']
    model = MLP(data_shape, hidden_size, classes_size)
    model.apply(init_param)
    return model