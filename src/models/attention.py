import torch
import torch.nn as nn
from config import cfg
from .utils import assist_loss_fn


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention = nn.Sequential(nn.Conv1d(input_size, hidden_size, 1, 1, 0),
                                       nn.ReLU(),
                                       nn.Conv1d(hidden_size, input_size, 1, 1, 0))

    def forward(self, input):
        output = {}
        x = input['output']
        attn = self.attention(x).softmax(-1)
        output['target'] = (x * attn).sum(-1)
        output = assist_loss_fn(input, output, self.training)
        return output


def attention():
    input_size = cfg['target_size']
    hidden_size = cfg['target_size']
    num_heads = cfg['attention']['num_heads']
    model = Attention(input_size, hidden_size, num_heads)
    return model
