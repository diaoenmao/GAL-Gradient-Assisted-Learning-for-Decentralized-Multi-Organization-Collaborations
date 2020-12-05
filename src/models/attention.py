import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param, loss_fn


class ScaledDotProduct(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, v, mask=None):
        scores = q.matmul(k.transpose(-2, -1)) / self.temperature
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        return output, attn


class Attention(nn.Module):
    def __init__(self, num_users, input_size, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.map_q = nn.Conv1d(input_size, hidden_size, 1, 1, 0)
        self.map_k = nn.Conv1d(input_size, hidden_size, 1, 1, 0)
        self.attention = ScaledDotProduct(temperature=(hidden_size // num_heads) ** 0.5)
        self.map_o = nn.Linear(num_users, 1)

    def _reshape_to_batches(self, x):
        batch_size, in_feature, seq_len = x.size()
        sub_dim = in_feature // self.num_heads
        return x.reshape(batch_size, self.num_heads, sub_dim, seq_len).permute(0, 1, 3, 2) \
            .reshape(batch_size * self.num_heads, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.num_heads
        out_dim = in_feature * self.num_heads
        return x.reshape(batch_size, self.num_heads, seq_len, in_feature).permute(0, 1, 3, 2) \
            .reshape(batch_size, out_dim, seq_len)

    def forward(self, input):
        output = {}
        x = input['output']
        q, k, v = x, x, x
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        q, attn = self.attention(q, k, v)
        q = self._reshape_from_batches(q)
        q = self.map_o(q).squeeze(-1)
        output['target'] = q
        if self.training:
            if input['assist'] is None:
                output['loss'] = loss_fn(output['target'], input['target'])
            else:
                output['loss'] = loss_fn(input['assist'] - cfg['assist_rate'] * output['target'], input['target'])
        return output


def attention():
    num_users = cfg['num_users']
    input_size = cfg['target_size']
    hidden_size = cfg['target_size']
    num_heads = cfg['attention']['num_heads']
    model = Attention(num_users, input_size, hidden_size, num_heads)
    model.apply(init_param)
    return model
