import datetime
import numpy as np
import sys
import time
import torch
import models
from config import cfg
from utils import to_device, make_optimizer, make_scheduler, collate


class Organization:
    def __init__(self, organization_id, feature_split, model_name):
        self.organization_id = organization_id
        self.feature_split = feature_split
        self.model_name = model_name
        self.model_parameters = [None for _ in range(cfg['global']['num_epochs'] + 1)]

    def initialize(self, dataset, metric, logger):
        input, output, initialization = {}, {}, {}
        train_target = torch.tensor(dataset['train'].target)
        test_target = torch.tensor(dataset['test'].target)
        if train_target.dtype == torch.int64:
            _, _, counts = torch.unique(train_target, sorted=True, return_inverse=True, return_counts=True)
            x = (counts / counts.sum()).log()
            initialization['train'] = x.view(1, -1).repeat(train_target.size(0), 1)
            initialization['test'] = x.view(1, -1).repeat(test_target.size(0), 1)
        else:
            x = train_target.mean()
            initialization['train'] = x.expand_as(train_target).detach().clone()
            initialization['test'] = x.expand_as(test_target).detach().clone()
        if 'train' in metric.metric_name:
            input['target'], output['target'] = train_target, initialization['train']
            output['loss'] = models.loss_fn(output['target'], input['target'])
            evaluation = metric.evaluate(metric.metric_name['train'], input, output)
            logger.append(evaluation, 'train', n=train_target.size(0))
        input['target'], output['target'] = test_target, initialization['test']
        output['loss'] = models.loss_fn(output['target'], input['target'])
        evaluation = metric.evaluate(metric.metric_name['test'], input, output)
        logger.append(evaluation, 'test', n=test_target.size(0))
        return initialization

    def train(self, iter, data_loader, metric, logger):
        model = eval('models.{}().to(cfg["device"])'.format(self.model_name[iter]))
        model.train(True)
        optimizer = make_optimizer(model, self.model_name[iter])
        scheduler = make_scheduler(optimizer, self.model_name[iter])
        for local_epoch in range(1, cfg[self.model_name[iter]]['num_epochs'] + 1):
            start_time = time.time()
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input['feature_split'] = self.feature_split
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                logger.append(evaluation, 'train', n=input_size)
            scheduler.step()
            local_time = (time.time() - start_time)
            local_finished_time = datetime.timedelta(
                seconds=round((cfg[self.model_name[iter]]['num_epochs'] - local_epoch) * local_time))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Local Epoch: {}({:.0f}%)'.format(local_epoch, 100. * local_epoch /
                                                                     cfg[self.model_name[iter]]['num_epochs']),
                             'ID: {}'.format(self.organization_id),
                             'Local Finished Time: {}'.format(local_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']), end='\r', flush=True)
        sys.stdout.write('\x1b[2K')
        self.model_parameters[iter] = model.to('cpu').state_dict()
        return

    def predict(self, iter, data_loader):
        with torch.no_grad():
            model = eval('models.{}().to(cfg["device"])'.format(self.model_name[iter]))
            model.load_state_dict(self.model_parameters[iter])
            model.train(False)
            organization_output = {'id': [], 'target': []}
            for i, input in enumerate(data_loader):
                input = collate(input)
                input['feature_split'] = self.feature_split
                input = to_device(input, cfg['device'])
                output = model(input)
                organization_output['id'].append(input['id'].cpu())
                output_target = output['target'].cpu()
                if cfg['noise'] > 0 and self.organization_id in cfg['noised_organization_id']:
                    noise = torch.normal(0, cfg['noise'], size=output_target.size())
                    output_target = output_target + noise
                organization_output['target'].append(output_target)
            organization_output['id'] = torch.cat(organization_output['id'], dim=0)
            organization_output['target'] = torch.cat(organization_output['target'], dim=0)
            organization_output['id'], indices = torch.sort(organization_output['id'])
            organization_output['target'] = organization_output['target'][indices]
        return organization_output
