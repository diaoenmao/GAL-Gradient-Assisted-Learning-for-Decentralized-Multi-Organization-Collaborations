import torch
import torch.nn as nn
import numpy as np
from config import cfg
from .utils import init_param, normalize, loss_fn, feature_split


class LSTM(nn.Module):
    def __init__(self, data_shape, hidden_size, num_layers, target_size):
        super().__init__()
        self.lstm = nn.LSTM(data_shape[0], hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.linear = nn.Linear(hidden_size, target_size)

    def feature(self, input):
        x = input['data']
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        x, _, _ = self.lstm(x)
        x = x[:, -1]
        return x

    def forward(self, input):
        output = {}
        x = input['data']
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        x, _, _ = self.lstm(x)
        x = x[:, -1]
        output['target'] = self.linear(x)
        output['loss'] = loss_fn(output['target'], input['target'])
        return output


def lstm():
    data_shape = cfg['data_shape']
    hidden_size = cfg['lstm']['hidden_size']
    num_layers = cfg['lstm']['num_layers']
    target_size = cfg['target_size']
    if cfg['assist_mode'] == 'interm':
        model = Interm(LSTM(data_shape, hidden_size, num_layers, target_size), hidden_size)
    elif cfg['assist_mode'] == 'late':
        model = Late(LSTM(data_shape, hidden_size, num_layers, target_size))
    elif cfg['assist_mode'] in ['bag', 'stack']:
        model = LSTM(data_shape, hidden_size, num_layers, target_size)
    else:
        raise ValueError('Not valid assist mode')
    model.apply(init_param)
    return model
