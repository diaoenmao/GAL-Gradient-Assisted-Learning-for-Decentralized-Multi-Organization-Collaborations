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
from data import fetch_dataset, split_dataset
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
cfg['metric_name'] = {'train': ['Loss', 'Loss_Local','Accuracy'], 'test': ['Loss', 'Accuracy']}


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
        last_epoch, assist, organization, logger = resume(cfg['model_tag'])
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, assist, organization, _ = resume(cfg['model_tag'])
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        feature_split = split_dataset(cfg['num_users'])
        assist = Assist(feature_split)
        organization = make_organization(assist)
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    if organization is None:
        organization = make_organization(assist)
    data_loader = assist.make_data_loader(dataset)
    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        logger.safe(True)
        train(data_loader, assist, organization, logger, epoch)
        organization_scores = broadcast(data_loader, organization, epoch)
        assist.update(data_loader, organization_scores)
        test(data_loader, assist, organization, logger, epoch)
        logger.safe(False)
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'assist': assist, 'organization': organization, 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] < logger.mean['test/{}'.format(cfg['pivot_metric'])]:
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(data_loader, assist, organization, logger, epoch):
    start_time = time.time()
    num_active_users = len(organization)
    for i in range(num_active_users):
        organization[i].train(epoch - 1, data_loader[i]['train'], logger, assist.organization_scores[i]['train'])
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


def test(data_loader, assist, organization, logger, epoch):
    with torch.no_grad():
        num_active_users = len(organization)
        for i in range(num_active_users):
            organization[i].test(epoch - 1, data_loader[i]['test'], logger, assist.organization_scores[i]['test'])
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test'])
    return


def broadcast(data_loader, organization, epoch):
    with torch.no_grad():
        num_active_users = len(organization)
        organization_scores = [{split: None for split in cfg['data_size']} for _ in range(num_active_users)]
        for i in range(num_active_users):
            for split in organization_scores[i]:
                organization_scores[i][split] = organization[i].broadcast(epoch - 1, data_loader[i][split])
    return organization_scores


def make_organization(assist):
    feature_split = assist.feature_split
    model_name = assist.make_model_name()
    organization = [None for _ in range(len(feature_split))]
    for i in range(len(feature_split)):
        model_name_i = model_name[i]
        feature_split_i = feature_split[i]
        organization[i] = Organization(i, feature_split_i, model_name_i)
    return organization


class Organization:
    def __init__(self, organization_id, feature_split, model_name):
        self.organization_id = organization_id
        self.feature_split = feature_split
        self.model_name = model_name
        self.model_parameters = [None for _ in range(cfg['num_epochs']['global'])]

    def train(self, iter, data_loader, logger, organization_scores=None):
        metric = Metric()
        model = eval('models.{}().to(cfg["device"])'.format(self.model_name))
        model.train(True)
        optimizer = make_optimizer(model)
        scheduler = make_scheduler(optimizer)
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
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
            if cfg['scheduler_name'] == 'ReduceLROnPlateau':
                scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
            else:
                scheduler.step()
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


def resume(model_tag, load_tag='checkpoint', verbose=True):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        checkpoint = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
        last_epoch = checkpoint['epoch']
        assist = checkpoint['assist']
        organization = checkpoint['organization']
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
        organization = None
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
    return last_epoch, assist, organization, logger


if __name__ == "__main__":
    main()