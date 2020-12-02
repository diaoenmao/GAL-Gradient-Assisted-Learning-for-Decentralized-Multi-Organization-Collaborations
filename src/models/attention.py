import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from .utils import init_param


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
        self.map_v = nn.Conv1d(input_size, hidden_size, 1, 1, 0)
        self.attention = ScaledDotProduct(temperature=(hidden_size // num_heads) ** 0.5)
        self.map_o = nn.Linear(num_users, 1)
        self.map_t = nn.Linear(num_users, num_users)

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
        x = input['score']
        q, k, v = self.map_q(x), self.map_k(x), x
        q, k, v = self._reshape_to_batches(q), self._reshape_to_batches(k), self._reshape_to_batches(v)
        q, attn = self.attention(q, k, v)
        q = self._reshape_from_batches(q)
        q = self.map_o(q).squeeze(-1)
        output['score'] = q
        if self.training:
            if input['assist'] is None:
                target = F.one_hot(input['label'], cfg['classes_size']).float()
                target[target == 0] = 1e-4
                target = torch.log(target)
                output['loss'] = F.mse_loss(output['score'], target)
            else:
                input['assist'].requires_grad = True
                loss = F.cross_entropy(input['assist'], input['label'], reduction='sum')
                loss.backward()
                target = copy.deepcopy(input['assist'].grad)
                output['loss'] = F.mse_loss(output['score'], target)
                input['assist'] = input['assist'].detach()
        return output


def attention():
    num_users = cfg['num_users']
    input_size = cfg['classes_size']
    hidden_size = cfg['classes_size']
    num_heads = 1
    model = Attention(num_users, input_size, hidden_size, num_heads)
    model.apply(init_param)
    return model