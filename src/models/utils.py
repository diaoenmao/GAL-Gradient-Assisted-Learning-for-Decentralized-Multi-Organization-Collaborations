import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


def init_param(m):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
    return m


def normalize(input):
    broadcast_size = [1] * input.dim()
    broadcast_size[1] = input.size(1)
    m, s = cfg['stats'][cfg['data_name']]
    m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
           torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
    input = input.sub(m).div(s)
    return input


def denormalize(input):
    broadcast_size = [1] * input.dim()
    broadcast_size[1] = input.size(1)
    m, s = cfg['stats'][cfg['data_name']]
    m, s = torch.tensor(m, dtype=input.dtype).view(broadcast_size).to(input.device), \
           torch.tensor(s, dtype=input.dtype).view(broadcast_size).to(input.device)
    input = input.mul(s).add(m)
    return input


def ce_loss(score, label):
    ce = F.cross_entropy(score, label)
    return ce


def kd_loss(score, label, assist):
    ce = F.cross_entropy(score, label)
    kld = nn.KLDivLoss(reduction='batchmean')
    log_p = F.log_softmax(score, dim=-1)
    kd = 0
    for i in range(assist.size(-1)):
        assist_i = assist[:, :, i]
        valid = assist_i.sum(dim=-1) != 0
        kd += kld(log_p[valid], F.softmax(assist_i[valid], dim=-1))
    kd = kd / assist.size(-1)
    alpha = 0.5
    kd = alpha * kd + (1 - alpha) * ce
    return kd