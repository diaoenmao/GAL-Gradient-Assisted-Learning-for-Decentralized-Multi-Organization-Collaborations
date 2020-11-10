import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import math
import models
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, load, to_device, process_dataset, resume, collate, save_img
from logger import Logger

# if __name__ == "__main__":
#     data_name = 'Wine'
#     subset = 'label'
#     dataset = fetch_dataset(data_name, subset)
#     process_dataset(dataset)
#     data_loader = make_data_loader(dataset)
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         print(input['feature'].size())
#         print(input[subset].size())
#         break
#     exit()

# if __name__ == "__main__":
#     N = 20
#     C = 5
#     score = torch.rand(N, C)
#     label = torch.randint(0, 2, (N,))
#     for i in range(10):
#         score.requires_grad = True
#         loss = F.cross_entropy(score, label)
#         print(loss)
#         loss.backward()
#         score = (score - score.grad).detach()
#
#
# if __name__ == "__main__":
#     # p = torch.tensor([0.1, 0.3, 0.2, 0.4]).view(1, -1)
#     # p = torch.tensor([1e-10, 1e-10, 1, 1e-10]).view(1, -1)
#     label = torch.tensor([2])
#     one_hot = torch.tensor([0, 0, 1, 0])
#     odds = one_hot.float()
#     odds[odds == 0] = 1e-4
#     log_odds = torch.log(odds)
#     sm = torch.softmax(log_odds, dim=-1)
#     loss = F.cross_entropy(log_odds.view(1, -1), label)
#     print(log_odds)
#     print(sm)
#     print(loss)