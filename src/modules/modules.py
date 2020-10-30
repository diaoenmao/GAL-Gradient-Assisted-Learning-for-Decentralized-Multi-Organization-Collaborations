import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuralController(nn.Module):
    def __init__(self, input_size, num_mode):
        super().__init__()
        self.input_size = input_size
        self.num_mode = num_mode
        codebook = self.make_codebook()
        self.register_buffer('codebook', codebook)

    def make_codebook(self):
        d = torch.distributions.bernoulli.Bernoulli(probs=0.5)
        codebook = set()
        while len(codebook) < self.num_mode:
            codebook_c = d.sample((self.num_mode, self.input_size))
            codebook_c = [tuple(c) for c in codebook_c.tolist()]
            codebook.update(codebook_c)
        codebook = torch.tensor(list(codebook)[:self.num_mode], dtype=torch.float)
        return codebook

    def forward(self, input):
        x, indicator = input
        code = indicator.matmul(self.codebook)
        code = code.view(*code.size(), *([1] * (x.dim() - 2)))
        output = [x * code.detach(), *input[1:]]
        return output


class Wrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, input):
        return [self.module(input[0]), *input[1:]]