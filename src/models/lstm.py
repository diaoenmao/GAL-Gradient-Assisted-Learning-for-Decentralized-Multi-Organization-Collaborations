import torch
import torch.nn as nn
import numpy as np
from config import cfg
from .utils import init_param, normalize, loss_fn, feature_split
from .interm import interm
from .late import late


class LSTM(nn.Module):
    def __init__(self, data_shape, hidden_size, num_layers, target_size):
        super().__init__()
        self.lstm = nn.LSTM(data_shape[0], hidden_size, num_layers, batch_first=True, dropout=0)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, target_size)

    def feature(self, input):
        x = input['data'][0]
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.dropout(x)
        return x

    def forward(self, input):
        output = {}
        x = input['data'][0]
        x = normalize(x)
        if 'feature_split' in input:
            x = feature_split(x, input['feature_split'])
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = self.dropout(x)
        x = self.linear(x)
        if cfg['data_name'] == 'MIMIC':
            output['target'] = x.unsqueeze(0)
        if 'target' in input:
            if 'loss_mode' in input:
                output['loss'] = loss_fn(output['target'], input['target'], loss_mode=input['loss_mode'])
            else:
                output['loss'] = loss_fn(output['target'], input['target'])
        return output


def lstm():
    data_shape = cfg['data_shape']
    hidden_size = cfg['lstm']['hidden_size']
    num_layers = cfg['lstm']['num_layers']
    target_size = cfg['target_size']
    if cfg['assist_mode'] == 'interm':
        model = interm(LSTM(data_shape, hidden_size, num_layers, target_size), hidden_size)
    elif cfg['assist_mode'] == 'late':
        model = late(LSTM(data_shape, hidden_size, num_layers, target_size))
    elif cfg['assist_mode'] in ['none', 'bag', 'stack']:
        model = LSTM(data_shape, hidden_size, num_layers, target_size)
    else:
        raise ValueError('Not valid assist mode')
    model.apply(init_param)
    return model
