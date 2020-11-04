import copy
import torch
import models
from config import cfg


class Assist:
    def __init__(self, feature_split):
        self.feature_split = feature_split

    def make_model_name(self):
        model_name = cfg['model_name'].split('-')
        model_idx = torch.randint(0, len(model_name), (len(self.feature_split),))
        model_name = [model_name[i] for i in model_idx]
        return model_name

    def make_model_parameters(self, model_name):
        model_parameters = [None for _ in range(len(self.feature_split))]
        for i in range(len(model_name)):
            model = eval('models.{}()'.format(model_name[i]))
            model_parameters[i] = copy.deepcopy(model.state_dict())
        return model_parameters