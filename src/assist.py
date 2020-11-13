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
        self.assist_parameters = [[None for _ in range(cfg[cfg['model_name']]['num_epochs']['global'])] for _ in
                                  range(len(self.feature_split))]

    def make_model_name(self):
        model_name = cfg['model_name'].split('-')
        model_idx = torch.randint(0, len(model_name), (len(self.feature_split),))
        model_name = [model_name[i] for i in model_idx]
        return model_name

    def make_data_loader(self, dataset):
        data_loader = [None for _ in range(len(self.feature_split))]
        for i in range(len(self.feature_split)):
            data_loader[i] = make_data_loader(dataset, cfg['model_name'])
        return data_loader

    def update(self, iter, data_loader, new_organization_scores):
        if cfg['assist_mode'] == 'none':
            for i in range(len(self.organization_scores)):
                for split in self.organization_scores[i]:
                    if self.organization_scores[i][split] is None:
                        self.organization_scores[i][split] = new_organization_scores[i][split]
                    else:
                        self.organization_scores[i][split]['score'] = self.organization_scores[i][split]['score'] - \
                                                                      self.assist_rate * \
                                                                      new_organization_scores[i][split]['score']
        elif cfg['assist_mode'] == 'bagging':
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
        elif cfg['assist_mode'] == 'stacking':
            _dataset = [{split: None for split in cfg['data_size']} for _ in range(len(self.feature_split))]
            for i in range(len(self.organization_scores)):
                for split in data_loader[i]:
                    assist = self.organization_scores[i][split]
                    score = []
                    for j in range(len(new_organization_scores)):
                        score.append(new_organization_scores[j][split]['score'])
                    score = torch.stack(score, dim=-1)
                    if assist is None:
                        _dataset[i][split] = torch.utils.data.TensorDataset(
                            torch.tensor(data_loader[i][split].dataset.id), score,
                            torch.tensor(data_loader[i][split].dataset.target))
                    else:
                        assist = assist['score']
                        _dataset[i][split] = torch.utils.data.TensorDataset(
                            torch.tensor(data_loader[i][split].dataset.id),
                            assist, score, torch.tensor(data_loader[i][split].dataset.target))
            for i in range(len(self.organization_scores)):
                _data_loader = make_data_loader(_dataset[i], 'assist')
                if 'train' in _data_loader:
                    model = models.stack().to(cfg['device'])
                    if iter > 0:
                        model.load_state_dict(self.assist_parameters[i][iter-1])
                    model.train(True)
                    optimizer = make_optimizer(model, 'assist')
                    scheduler = make_scheduler(optimizer, 'assist')
                    for assist_epoch in range(1, cfg['assist']['num_epochs']['global'] + 1):
                        for j, input in enumerate(_data_loader['train']):
                            if len(input) == 3:
                                input = {'id': input[0], 'assist': None, 'score': input[1], 'label': input[2]}
                            else:
                                input = {'id': input[0], 'assist': input[1], 'score': input[2], 'label': input[3]}
                            input = to_device(input, cfg['device'])
                            optimizer.zero_grad()
                            output = model(input)
                            output['loss'].backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                            optimizer.step()
                        scheduler.step()
                    self.assist_parameters[i][iter] = model.to('cpu').state_dict()
                with torch.no_grad():
                    for split in _data_loader:
                        organization_scores = {'id': torch.arange(cfg['data_size'][split]),
                                               'score': torch.zeros(cfg['data_size'][split], cfg['classes_size'])}
                        model = models.stack().to(cfg['device'])
                        model.load_state_dict(self.assist_parameters[i][iter])
                        model.train(False)
                        for j, input in enumerate(_data_loader[split]):
                            if len(input) == 3:
                                input = {'id': input[0], 'assist': None, 'score': input[1]}
                            else:
                                input = {'id': input[0], 'assist': input[1], 'score': input[2]}
                            input = to_device(input, cfg['device'])
                            output = model(input)
                            organization_scores['score'][input['id']] = output['score'].cpu()
                        if self.organization_scores[i][split] is None:
                            self.organization_scores[i][split] = organization_scores
                        else:
                            self.organization_scores[i][split]['score'] = self.organization_scores[i][split]['score'] \
                                                                          - self.assist_rate * \
                                                                          organization_scores['score']
        elif cfg['assist_mode'] == 'attention':
            _dataset = [{split: None for split in cfg['data_size']} for _ in range(len(self.feature_split))]
            for i in range(len(self.organization_scores)):
                for split in data_loader[i]:
                    assist = self.organization_scores[i][split]
                    score = []
                    for j in range(len(new_organization_scores)):
                        score.append(new_organization_scores[j][split]['score'])
                    score = torch.stack(score, dim=-1)
                    if assist is None:
                        _dataset[i][split] = torch.utils.data.TensorDataset(
                            torch.tensor(data_loader[i][split].dataset.id), score,
                            torch.tensor(data_loader[i][split].dataset.target))
                    else:
                        assist = assist['score']
                        _dataset[i][split] = torch.utils.data.TensorDataset(
                            torch.tensor(data_loader[i][split].dataset.id),
                            assist, score, torch.tensor(data_loader[i][split].dataset.target))
            for i in range(len(self.organization_scores)):
                _data_loader = make_data_loader(_dataset[i], 'assist')
                if 'train' in _data_loader:
                    model = models.attention().to(cfg['device'])
                    if iter > 0:
                        model.load_state_dict(self.assist_parameters[i][iter-1])
                    model.train(True)
                    optimizer = make_optimizer(model, 'assist')
                    scheduler = make_scheduler(optimizer, 'assist')
                    for assist_epoch in range(1, cfg['assist']['num_epochs']['global'] + 1):
                        for j, input in enumerate(_data_loader['train']):
                            if len(input) == 3:
                                input = {'id': input[0], 'assist': None, 'score': input[1], 'label': input[2]}
                            else:
                                input = {'id': input[0], 'assist': input[1], 'score': input[2], 'label': input[3]}
                            input = to_device(input, cfg['device'])
                            optimizer.zero_grad()
                            output = model(input)
                            output['loss'].backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                            optimizer.step()
                        scheduler.step()
                    self.assist_parameters[i][iter] = model.to('cpu').state_dict()
                with torch.no_grad():
                    for split in _data_loader:
                        organization_scores = {'id': torch.arange(cfg['data_size'][split]),
                                               'score': torch.zeros(cfg['data_size'][split], cfg['classes_size'])}
                        model = models.attention().to(cfg['device'])
                        model.load_state_dict(self.assist_parameters[i][iter])
                        model.train(False)
                        for j, input in enumerate(_data_loader[split]):
                            if len(input) == 3:
                                input = {'id': input[0], 'assist': None, 'score': input[1]}
                            else:
                                input = {'id': input[0], 'assist': input[1], 'score': input[2]}
                            input = to_device(input, cfg['device'])
                            output = model(input)
                            organization_scores['score'][input['id']] = output['score'].cpu()
                        if self.organization_scores[i][split] is None:
                            self.organization_scores[i][split] = organization_scores
                        else:
                            self.organization_scores[i][split]['score'] = self.organization_scores[i][split]['score'] \
                                                                          - self.assist_rate * \
                                                                          organization_scores['score']
        else:
            raise ValueError('Not valid assist')
        return