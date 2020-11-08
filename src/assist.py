import copy
import torch
import models
from config import cfg
from data import make_data_loader


class Assist:
    def __init__(self, feature_split):
        self.feature_split = feature_split
        self.organization_scores = {split: None for split in cfg['data_size']}
        self.assist_rate = cfg['assist_rate']

    def make_model_name(self):
        model_name = cfg['model_name'].split('-')
        model_idx = torch.randint(0, len(model_name), (len(self.feature_split),))
        model_name = [model_name[i] for i in model_idx]
        return model_name

    def make_data_loader(self, dataset):
        data_loader = [None for _ in range(len(self.feature_split))]
        for i in range(len(self.feature_split)):
            data_loader[i] = make_data_loader(dataset)
        return data_loader

    def update(self, data_loader, new_organization_scores):
        organization_scores = {split: {'id': torch.arange(cfg['data_size'][split]),
                                       'score': torch.zeros(cfg['data_size'][split], cfg['classes_size'])}
                               for split in cfg['data_size']}
        mask = {split: torch.zeros(cfg['data_size'][split], cfg['classes_size']) for split in cfg['data_size']}
        for split in cfg['data_size']:
            for i in range(len(new_organization_scores)):
                id = new_organization_scores[i][split]['id']
                mask[split][id] += 1
                organization_scores[split]['score'][id] += new_organization_scores[i][split]['score']
            organization_scores[split]['score'] = organization_scores[split]['score'] / mask[split]
        for split in self.organization_scores:
            if self.organization_scores[split] is None:
                self.organization_scores[split] = organization_scores[split]
            else:
                self.organization_scores[split]['score'] = self.organization_scores[split]['score'] - \
                                                           self.assist_rate * \
                                                           organization_scores[split]['score']
        return