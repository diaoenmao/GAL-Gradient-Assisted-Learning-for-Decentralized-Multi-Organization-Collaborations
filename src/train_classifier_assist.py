import argparse
import copy
import datetime
import models
import os
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset
from assist import Assist
from metrics import Metric
from utils import save, load, to_device, process_control, process_dataset, make_optimizer, make_scheduler, collate
from logger import Logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']]) if 'control' in cfg else ''
cfg['pivot_metric'] = 'Accuracy'
cfg['pivot'] = -float('inf')
cfg['metric_name'] = {'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']}


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset)
    if cfg['resume_mode'] == 1:
        last_epoch, assist, logger = resume(cfg['model_tag'])
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, assist, _ = resume(cfg['model_tag'])
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        feature_split = split_dataset(cfg['num_users'])
        assist = Assist(feature_split)
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    organization = make_organization(dataset, assist)
    init_train(organization, logger, 0)
    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        logger.safe(True)
        train(organization, logger, epoch)
        test(organization, logger, epoch)
        logger.safe(False)
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'assist': assist, 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] < logger.mean['test/{}'.format(cfg['pivot_metric'])]:
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def init_train(organization, logger, epoch):
    num_active_users = len(organization)
    start_time = time.time()
    for i in range(num_active_users):
        organization[i].train(logger)
        if i % int((num_active_users * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / num_active_users),
                             'ID: {}/{}'.format(i + 1, num_active_users),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            logger.write('train', cfg['metric_name']['train'])
    return


def train(organization, logger, epoch):
    start_time = time.time()
    num_active_users = len(organization)
    organization_scores = [None for _ in range(num_active_users)]
    for i in range(num_active_users):
        organization_scores[i] = organization[i].broadcast()
    for i in range(num_active_users):
        other_scores = organization_scores[:i] + organization_scores[i + 1:]
        organization[i].train(logger, other_scores)
        if i % int((num_active_users * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / num_active_users),
                             'ID: {}/{}'.format(i + 1, num_active_users),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            logger.write('train', cfg['metric_name']['train'])
    return


def test(organization, logger, epoch):
    with torch.no_grad():
        num_active_users = len(organization)
        organization_scores = [None for _ in range(num_active_users)]
        for i in range(num_active_users):
            organization_scores[i] = organization[i].broadcast()
        for i in range(num_active_users):
            other_scores = organization_scores[:i] + organization_scores[i + 1:]
            organization[i].test(logger, other_scores)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test'])
    return


def make_organization(dataset, assist):
    feature_split = assist.feature_split
    num_active_users = len(feature_split)
    model_name = assist.make_model_name()
    model_parameters = assist.make_model_parameters(model_name)
    organization = [None for _ in range(len(feature_split))]
    for i in range(num_active_users):
        data_loader_i = make_data_loader(dataset)
        model_name_i = model_name[i]
        model_parameters_i = model_parameters[i]
        feature_split_i = feature_split[i]
        organization[i] = Organization(data_loader_i, model_name_i, model_parameters_i, feature_split_i)
    return organization


class Organization:
    def __init__(self, data_loader, model_name, model_parameters, feature_split):
        self.data_loader = data_loader
        self.model_name = model_name
        self.model_parameters = model_parameters
        self.feature_split = feature_split

    def train(self, logger, other_scores=None):
        metric = Metric()
        model = eval('models.{}()'.format(self.model_name))
        model.load_state_dict(self.model_parameters)
        model = model.to(cfg['device'])
        model.train(True)
        optimizer = make_optimizer(model)
        scheduler = make_scheduler(optimizer)
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for i, input in enumerate(self.data_loader['train']):
                input = collate(input)
                input_size = input[cfg['data_tag']].size(0)
                input['feature_split'] = self.feature_split
                if other_scores is not None:
                    input['assist'] = self.align(input['id'], other_scores)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(cfg['metric_name']['train'], input, output)
                logger.append(evaluation, 'train', n=input_size)
            if cfg['scheduler_name'] == 'ReduceLROnPlateau':
                scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
            else:
                scheduler.step()
        self.model_parameters = model.to('cpu').state_dict()
        return

    def test(self, logger, other_scores=None):
        with torch.no_grad():
            metric = Metric()
            model = eval('models.{}()'.format(self.model_name))
            model.load_state_dict(self.model_parameters)
            model = model.to(cfg['device'])
            model.train(False)
            for i, input in enumerate(self.data_loader['test']):
                input = collate(input)
                input_size = input[cfg['data_tag']].size(0)
                input['feature_split'] = self.feature_split
                if other_scores is not None:
                    input['assist'] = self.align(input['id'], other_scores)
                input = to_device(input, cfg['device'])
                output = model(input)
                evaluation = metric.evaluate(cfg['metric_name']['test'], input, output)
                logger.append(evaluation, 'test', n=input_size)
        return

    def broadcast(self):
        with torch.no_grad():
            model = eval('models.{}()'.format(self.model_name))
            model.load_state_dict(self.model_parameters)
            model = model.to(cfg['device'])
            model.train(False)
            organization_scores = {'id': [], 'score': []}
            for i, input in enumerate(self.data_loader['train']):
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

    def align(self, id, other_scores):
        assist = []
        for i in range(len(other_scores)):
            assist_i = torch.zeros(id.size(0), cfg['classes_size'])
            _, id_idx, organization_idx = np.intersect1d(id.numpy(), other_scores[i]['id'].numpy(),
                                                         assume_unique=True, return_indices=True)
            assist_i[torch.tensor(id_idx)] = other_scores[i]['score'][torch.tensor(organization_idx)]
            assist.append(assist_i)
        assist = torch.stack(assist, dim=-1)
        return assist


def resume(model_tag, load_tag='checkpoint', verbose=True):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        checkpoint = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
        last_epoch = checkpoint['epoch']
        assist = checkpoint['assist']
        logger = checkpoint['logger']
        if verbose:
            print('Resume from {}'.format(last_epoch))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        from datetime import datetime
        from logger import Logger
        last_epoch = 1
        feature_split = split_dataset(cfg['num_users'])
        assist = Assist(feature_split)
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
    return last_epoch, assist, logger


if __name__ == "__main__":
    main()