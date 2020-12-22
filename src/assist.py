import copy
import torch
import models
from config import cfg
from data import make_data_loader
from organization import Organization
from utils import make_optimizer, make_scheduler, collate, to_device


class Assist:
    def __init__(self, feature_split):
        self.feature_split = feature_split
        self.model_name = self.make_model_name()
        self.assist_parameters = [[None for _ in range(cfg['global']['num_epochs'])] for _ in
                                  range(len(self.feature_split))]
        self.assist_rates = [[None for _ in range(cfg['global']['num_epochs'])] for _ in
                             range(len(self.feature_split))]
        self.reset()

    def reset(self):
        self.organization_outputs = [{split: None for split in cfg['data_size']} for _ in
                                     range(len(self.feature_split))]
        return

    def make_model_name(self):
        model_name = cfg['model_name'].split('-')
        model_idx = torch.arange(len(model_name)).repeat(round(len(self.feature_split) / len(model_name)))
        model_idx = model_idx.tolist()[:len(self.feature_split)]
        model_name = [model_name[i] for i in model_idx]
        return model_name

    def make_data_loader(self, dataset):
        data_loader = [None for _ in range(len(self.feature_split))]
        for i in range(len(self.feature_split)):
            data_loader[i] = make_data_loader(dataset, self.model_name[i])
        return data_loader

    def make_organization(self):
        feature_split = self.feature_split
        model_name = self.model_name
        organization = [None for _ in range(len(feature_split))]
        for i in range(len(feature_split)):
            model_name_i = model_name[i]
            feature_split_i = feature_split[i]
            organization[i] = Organization(i, feature_split_i, model_name_i)
        return organization

    def update(self, iter, data_loader, new_organization_outputs):
        import time
        if cfg['assist_mode'] == 'none':
            with torch.no_grad():
                organization_outputs = [{split: {'id': torch.arange(cfg['data_size'][split]),
                                                 'target': torch.zeros(cfg['data_size'][split], cfg['target_size'])}
                                         for split in data_loader[0]} for _ in range(len(self.feature_split))]
                for i in range(len(self.feature_split)):
                    for split in data_loader[0]:
                        organization_outputs[i][split]['target'] = new_organization_outputs[i][split]['target']
        elif cfg['assist_mode'] == 'bag':
            _new_organization_outputs = {split: {'target': []} for split in data_loader[0]}
            for split in data_loader[0]:
                for i in range(len(new_organization_outputs)):
                    _new_organization_outputs[split]['target'].append(new_organization_outputs[i][split]['target'])
                _new_organization_outputs[split]['target'] = torch.stack(_new_organization_outputs[split]['target'],
                                                                         dim=-1)
            with torch.no_grad():
                organization_outputs = [{split: {'id': torch.arange(cfg['data_size'][split]),
                                                 'target': torch.zeros(cfg['data_size'][split], cfg['target_size'])}
                                         for split in data_loader[0]} for _ in range(len(self.feature_split))]
                for i in range(len(self.feature_split)):
                    for split in data_loader[0]:
                        organization_outputs[i][split]['target'] = _new_organization_outputs[split]['target'].mean(
                            dim=-1)
        elif cfg['assist_mode'] in ['stack']:
            _new_organization_outputs = {split: {'target': []} for split in data_loader[0]}
            for split in data_loader[0]:
                for i in range(len(new_organization_outputs)):
                    _new_organization_outputs[split]['target'].append(new_organization_outputs[i][split]['target'])
                _new_organization_outputs[split]['target'] = torch.stack(_new_organization_outputs[split]['target'],
                                                                         dim=-1)
            _dataset = [{split: None for split in data_loader[0]} for _ in range(len(self.feature_split))]
            for i in range(len(self.feature_split)):
                for split in data_loader[i]:
                    if self.organization_outputs[i][split] is None:
                        _dataset[i][split] = torch.utils.data.TensorDataset(
                            torch.tensor(data_loader[i][split].dataset.id), _new_organization_outputs[split]['target'],
                            torch.tensor(data_loader[i][split].dataset.target))
                    else:
                        _dataset[i][split] = torch.utils.data.TensorDataset(
                            torch.tensor(data_loader[i][split].dataset.id),
                            _new_organization_outputs[split]['target'],
                            torch.tensor(data_loader[i][split].dataset.target),
                            self.organization_outputs[i][split]['target'])
            for i in range(len(self.feature_split)):
                if 'train' in _dataset[i]:
                    tensors = _dataset[i]['train'].tensors
                    input = {'id': tensors[0], 'output': tensors[1], 'target': tensors[2], 'assist': None} if len(
                        tensors) == 3 else {'id': tensors[0], 'output': tensors[1], 'target': tensors[2],
                                            'assist': tensors[3]}
                    input = to_device(input, cfg['device'])
                    model = eval('models.{}().to(cfg["device"])'.format(cfg['assist_mode']))
                    model.train(True)
                    optimizer = make_optimizer(model, 'assist')
                    for assist_epoch in range(1, cfg['assist']['num_epochs'] + 1):
                        def closure():
                            output = model(input)
                            optimizer.zero_grad()
                            output['loss'].backward()
                            return output['loss']

                        optimizer.step(closure)
                    self.assist_parameters[i][iter] = model.to('cpu').state_dict()
            with torch.no_grad():
                organization_outputs = [{split: {'id': torch.arange(cfg['data_size'][split]),
                                                 'target': torch.zeros(cfg['data_size'][split], cfg['target_size'])}
                                         for split in data_loader[0]} for _ in range(len(self.feature_split))]
                for i in range(len(self.feature_split)):
                    _data_loader = make_data_loader(_dataset[i], 'assist', {'train': False, 'test': False})
                    for split in data_loader[0]:
                        model = eval('models.{}().to(cfg["device"])'.format(cfg['assist_mode']))
                        model.load_state_dict(self.assist_parameters[i][iter])
                        model.train(False)
                        output = []
                        for j, input in enumerate(_data_loader[split]):
                            input = {'id': input[0], 'output': input[1], 'target': input[2], 'assist': None} if len(
                                input) == 3 else {'id': input[0], 'output': input[1], 'target': input[2],
                                                  'assist': input[3]}
                            input = to_device(input, cfg['device'])
                            output = model(input)
                            organization_outputs[i][split]['target'][input['id']] = output['target'][:].cpu()
        else:
            raise ValueError('Not valid assist')
        if self.organization_outputs[i][split] is not None and 'train' in data_loader[0]:
            for i in range(len(self.feature_split)):
                input = {'id': torch.tensor(data_loader[i]['train'].dataset.id),
                         'output': organization_outputs[i]['train']['target'],
                         'target': torch.tensor(data_loader[i]['train'].dataset.target),
                         'assist': self.organization_outputs[i]['train']['target']}
                input = to_device(input, cfg['device'])
                model = models.linesearch().to(cfg['device'])
                model.train(True)
                optimizer = make_optimizer(model, 'linesearch')
                for linearsearch_epoch in range(1, cfg['linesearch']['num_epochs'] + 1):
                    def closure():
                        output = model(input)
                        optimizer.zero_grad()
                        output['loss'].backward()
                        return output['loss']

                    optimizer.step(closure)
                self.assist_rates[i][iter] = model.assist_rate.item()
        with torch.no_grad():
            for i in range(len(self.organization_outputs)):
                for split in data_loader[i]:
                    if self.organization_outputs[i][split] is None:
                        self.organization_outputs[i][split] = copy.deepcopy(organization_outputs[i][split])
                    else:
                        self.organization_outputs[i][split]['target'] = self.organization_outputs[i][split][
                                                                            'target'] + self.assist_rates[i][iter] * \
                                                                        organization_outputs[i][split]['target']
        return
