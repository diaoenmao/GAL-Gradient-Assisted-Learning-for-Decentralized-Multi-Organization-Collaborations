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

if __name__ == "__main__":
    data_name = 'Wine'
    subset = 'label'
    dataset = fetch_dataset(data_name, subset)
    process_dataset(dataset)
    data_loader = make_data_loader(dataset)
    for i, input in enumerate(data_loader['train']):
        input = collate(input)
        print(input['feature'].size())
        print(input[subset].size())
        break
    exit()