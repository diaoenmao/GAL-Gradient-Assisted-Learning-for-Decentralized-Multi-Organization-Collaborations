import copy
import torch
import models
from config import cfg
from data import make_data_loader
from utils import make_optimizer, make_scheduler, collate, to_device


class Assist:
    def __init__(self, feature_split):
        self.feature_split = feature_split
        self.organization_scores = [{split: None for split in cfg['data_size']} for _ in range(len(self.feature_split))]
        self.assist_rate = cfg['assist_rate']
        self.assist_parameters = [None for _ in range(cfg['num_epochs']['global'])]

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

    def update(self, iter, dataset, new_organization_scores):
        if cfg['assist'] == 'none':
            for i in range(len(self.organization_scores)):
                for split in self.organization_scores[i]:
                    if self.organization_scores[i][split] is None:
                        self.organization_scores[i][split] = new_organization_scores[i][split]
                    else:
                        self.organization_scores[i][split]['score'] = self.organization_scores[i][split]['score'] - \
                                                                      self.assist_rate * \
                                                                      new_organization_scores[i][split]['score']
        elif cfg['assist'] == 'bagging':
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
            for i in range(len(self.organization_scores)):
                for split in self.organization_scores[i]:
                    if self.organization_scores[i][split] is None:
                        self.organization_scores[i][split] = organization_scores[split]
                    else:
                        self.organization_scores[i][split]['score'] = self.organization_scores[i][split]['score'] - \
                                                                      self.assist_rate * \
                                                                      organization_scores[split]['score']
        elif cfg['assist'] == 'stacking':
            _dataset = {}
            for split in dataset:
                for i in range(len(self.organization_scores)):
                    assist = self.organization_scores[i][split]
                    scores = []
                    for j in range(len(new_organization_scores)):
                        scores.append(new_organization_scores[j][split])
                    scores = torch.stack(scores, dim=-1)
                    _dataset[split] = torch.utils.data.TensorDataset(assist, scores,
                                                                     torch.tensor(dataset[split].target))
            _data_loader = make_data_loader(dataset)
            if 'train' in _data_loader:
                model = models.stacker().to(cfg['device'])
                model.train(True)
                optimizer = make_optimizer(model)
                scheduler = make_scheduler(optimizer)
                for assist_epoch in range(1, cfg['num_epochs']['assist'] + 1):
                    for i, input in enumerate(_data_loader['train']):
                        input = {'assist': input[0], 'scores': input[1], 'label': input[2]}
                        input = to_device(input, cfg['device'])
                        optimizer.zero_grad()
                        output = model(input)
                        output['loss'].backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        optimizer.step()
                    scheduler.step()
                self.assist_parameters[iter] = model.to('cpu').state_dict()

        elif cfg['assist'] == 'attention':
            pass
        else:
            raise ValueError('Not valid assist')
        return