import copy
import torch
import models
from config import cfg
from data import make_data_loader
from organization import Organization
from utils import make_optimizer, to_device


class Assist:
    def __init__(self, feature_split):
        self.feature_split = feature_split
        self.model_name = self.make_model_name()
        self.assist_parameters = [None for _ in range(cfg['global']['num_epochs'] + 1)]
        self.assist_rates = [None for _ in range(cfg['global']['num_epochs'] + 1)]
        self.reset()

    def reset(self):
        self.organization_output = [{split: None for split in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        self.organization_target = [{split: None for split in cfg['data_size']} for _ in
                                    range(cfg['global']['num_epochs'] + 1)]
        return

    def make_model_name(self):
        model_name_list = cfg['model_name'].split('-')
        if len(model_name_list) == 1:
            model_name = [model_name_list[0] for _ in range(cfg['global']['num_epochs'] + 1)]
            model_name = [model_name for _ in range(len(self.feature_split))]
        elif len(model_name_list) == 2:
            model_name = [model_name_list[0]] + [model_name_list[1] for _ in range(cfg['global']['num_epochs'])]
            model_name = [model_name for _ in range(len(self.feature_split))]
        else:
            raise ValueError('Not valid model name')
        return model_name

    def make_organization(self):
        feature_split = self.feature_split
        model_name = self.model_name
        organization = [None for _ in range(len(feature_split))]
        for i in range(len(feature_split)):
            model_name_i = model_name[i]
            feature_split_i = feature_split[i]
            organization[i] = Organization(i, feature_split_i, model_name_i)
        return organization

    def broadcast(self, dataset, iter):
        for split in dataset:
            self.organization_output[iter - 1][split].requires_grad = True
            loss = models.loss_fn(self.organization_output[iter - 1][split],
                                  self.organization_target[0][split], reduction='sum')
            loss.backward()
            self.organization_target[iter][split] = - copy.deepcopy(self.organization_output[iter - 1][split].grad)
            dataset[split].target = self.organization_target[iter][split].numpy()
            self.organization_output[iter - 1][split].detach_()
        data_loader = [None for _ in range(len(self.feature_split))]
        for i in range(len(self.feature_split)):
            data_loader[i] = make_data_loader(dataset, self.model_name[i][iter])
        return data_loader

    def update(self, organization_outputs, iter):
        if cfg['assist_mode'] == 'none':
            for split in organization_outputs[0]:
                self.organization_output[iter][split] = organization_outputs[0][split]
        elif cfg['assist_mode'] == 'bag':
            _organization_outputs = {split: [] for split in organization_outputs[0]}
            for split in organization_outputs[0]:
                for i in range(len(organization_outputs)):
                    _organization_outputs[split].append(organization_outputs[i][split])
                _organization_outputs[split] = torch.stack(_organization_outputs[split], dim=-1)
            for split in organization_outputs[0]:
                self.organization_output[iter][split] = _organization_outputs[split].mean(dim=-1)
        elif cfg['assist_mode'] == 'stack':
            _organization_outputs = {split: [] for split in organization_outputs[0]}
            for split in organization_outputs[0]:
                for i in range(len(organization_outputs)):
                    _organization_outputs[split].append(organization_outputs[i][split])
                _organization_outputs[split] = torch.stack(_organization_outputs[split], dim=-1)
            if 'train' in organization_outputs[0]:
                input = {'output': _organization_outputs['train'],
                         'target': self.organization_target[iter]['train']}
                input = to_device(input, cfg['device'])
                model = eval('models.{}().to(cfg["device"])'.format(cfg['assist_mode']))
                model.train(True)
                optimizer = make_optimizer(model, 'assist')
                for assist_epoch in range(1, cfg['assist']['num_epochs'] + 1):
                    output = model(input)
                    optimizer.zero_grad()
                    output['loss'].backward()
                    optimizer.step()
                self.assist_parameters[iter] = model.to('cpu').state_dict()
            with torch.no_grad():
                model = eval('models.{}().to(cfg["device"])'.format(cfg['assist_mode']))
                model.load_state_dict(self.assist_parameters[iter])
                model.train(False)
                for split in organization_outputs[0]:
                    input = {'output': _organization_outputs[split],
                             'target': self.organization_target[iter][split]}
                    input = to_device(input, cfg['device'])
                    self.organization_output[iter][split] = model(input)['target'].cpu()
        else:
            raise ValueError('Not valid assist')
        if 'train' in organization_outputs[0]:
            if cfg['assist_rate_mode'] == 'search':
                input = {'history': self.organization_output[iter - 1]['train'],
                         'output': self.organization_output[iter]['train'],
                         'target': self.organization_target[0]['train']}
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
                self.assist_rates[iter] = model.assist_rate.item()
            else:
                self.assist_rates[iter] = cfg['linesearch']['lr']
        with torch.no_grad():
            for split in organization_outputs[0]:
                self.organization_output[iter][split] = self.organization_output[iter - 1][split] + self.assist_rates[
                    iter] * self.organization_output[iter][split]
        return
