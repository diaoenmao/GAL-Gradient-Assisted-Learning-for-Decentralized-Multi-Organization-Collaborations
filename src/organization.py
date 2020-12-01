import numpy as np
import torch
import models
from config import cfg
from metrics import Metric
from utils import to_device, make_optimizer, make_scheduler, collate


class Organization:
    def __init__(self, organization_id, feature_split, model_name):
        self.organization_id = organization_id
        self.feature_split = feature_split
        self.model_name = model_name
        self.model_parameters = [None for _ in range(cfg['global']['num_epochs'])]

    def train(self, iter, data_loader, logger, organization_scores=None):
        metric = Metric()
        model = eval('models.{}().to(cfg["device"])'.format(self.model_name))
        model.train(True)
        optimizer = make_optimizer(model, self.model_name)
        scheduler = make_scheduler(optimizer, self.model_name)
        for local_epoch in range(1, cfg[self.model_name]['num_epochs'] + 1):
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input[cfg['data_tag']].size(0)
                input['feature_split'] = self.feature_split
                input['assist'] = None if organization_scores is None else self.align(input['id'], organization_scores)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss_local'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(cfg['metric_name']['train'], input, output)
                logger.append(evaluation, 'train', n=input_size)
            if cfg[self.model_name]['scheduler_name'] == 'ReduceLROnPlateau':
                scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
            else:
                scheduler.step()
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Local Epoch: {}({:.0f}%)'.format(local_epoch, 100. * local_epoch /
                                                                     cfg[self.model_name]['num_epochs']),
                             'ID: {}'.format(self.organization_id)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', cfg['metric_name']['train']), end='\r', flush=True)
        self.model_parameters[iter] = model.to('cpu').state_dict()
        return

    def test(self, iter, data_loader, logger, organization_scores=None):
        with torch.no_grad():
            metric = Metric()
            model = eval('models.{}().to(cfg["device"])'.format(self.model_name))
            model.load_state_dict(self.model_parameters[iter])
            model.train(False)
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input[cfg['data_tag']].size(0)
                input['feature_split'] = self.feature_split
                input['assist'] = None if organization_scores is None else self.align(input['id'], organization_scores)
                input = to_device(input, cfg['device'])
                output = model(input)
                evaluation = metric.evaluate(cfg['metric_name']['test'], input, output)
                logger.append(evaluation, 'test', n=input_size)
        return

    def broadcast(self, iter, data_loader):
        with torch.no_grad():
            model = eval('models.{}().to(cfg["device"])'.format(self.model_name))
            model.load_state_dict(self.model_parameters[iter])
            model.train(False)
            organization_scores = {'id': [], 'score': []}
            for i, input in enumerate(data_loader):
                input = collate(input)
                input['feature_split'] = self.feature_split
                input = to_device(input, cfg['device'])
                output = model(input)
                organization_scores['id'].append(input['id'].cpu())
                organization_scores['score'].append(output['score'].cpu())
            organization_scores['id'] = torch.cat(organization_scores['id'], dim=0)
            organization_scores['score'] = torch.cat(organization_scores['score'], dim=0)
            organization_scores['id'], indices = torch.sort(organization_scores['id'])
            organization_scores['score'] = organization_scores['score'][indices]
        return organization_scores

    def align(self, id, organization_scores):
        assist = torch.empty(id.size(0), cfg['classes_size']).fill_(float('nan'))
        _, id_idx, organization_idx = np.intersect1d(id.numpy(), organization_scores['id'].numpy(),
                                                     assume_unique=True, return_indices=True)
        assist[torch.tensor(id_idx)] = organization_scores['score'][torch.tensor(organization_idx)]
        return assist