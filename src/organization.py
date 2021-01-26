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
            if cfg[self.model_name[iter]]['scheduler_name'] == 'ReduceLROnPlateau':
                scheduler.step(metrics=logger.mean['train/{}'.format(metric.pivot_name)])
            else:
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

    def test(self, iter, data_loader, metric, logger):
        with torch.no_grad():
            model = eval('models.{}().to(cfg["device"])'.format(self.model_name[iter]))
            model.load_state_dict(self.model_parameters[iter])
            model.train(False)
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                input['feature_split'] = self.feature_split
                input = to_device(input, cfg['device'])
                output = model(input)
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                logger.append(evaluation, 'test', n=input_size)
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
                organization_output['target'].append(output['target'].cpu())
            organization_output['id'] = torch.cat(organization_output['id'], dim=0)
            organization_output['target'] = torch.cat(organization_output['target'], dim=0)
            organization_output['id'], indices = torch.sort(organization_output['id'])
            organization_output['target'] = organization_output['target'][indices]
        return organization_output
