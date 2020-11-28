import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param, normalize, feature_split


class Conv(nn.Module):
    def __init__(self, data_shape, hidden_size, classes_size):
        super().__init__()
        blocks = [nn.Conv2d(data_shape[0], hidden_size[0], 3, 1, 1),
                  nn.BatchNorm2d(hidden_size[0]),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        for i in range(len(hidden_size) - 1):
            blocks.extend([nn.Conv2d(hidden_size[i], hidden_size[i + 1], 3, 1, 1),
                           nn.BatchNorm2d(hidden_size[i + 1]),
                           nn.ReLU(inplace=True),
                           nn.MaxPool2d(2)])
        blocks = blocks[:-1]
        blocks.extend([nn.AdaptiveAvgPool2d(1),
                       nn.Flatten(),
                       nn.Linear(hidden_size[-1], classes_size)])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        output = {'loss': torch.tensor(0, device=cfg['device'], dtype=torch.float32)}
        x = input[cfg['data_tag']]
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        out = self.blocks(x)
        output['score'] = out
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


def conv():
    data_shape = cfg['data_shape']
    hidden_size = cfg['conv']['hidden_size']
    classes_size = cfg['classes_size']
    model = Conv(data_shape, hidden_size, classes_size)
    model.apply(init_param)
    return model