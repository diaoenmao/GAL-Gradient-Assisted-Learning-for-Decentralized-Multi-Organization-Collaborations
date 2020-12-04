import argparse
import copy
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data import fetch_dataset, split_dataset
from metrics import Metric
from assist import Assist
from utils import save, load, process_control, process_dataset
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


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'])
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
        feature_split = split_dataset(cfg['num_users'], cfg['feature_split_mode'])
        assist = Assist(feature_split, cfg['assist_rate'])
        organization = assist.make_organization()
        current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], current_time)
        logger = Logger(logger_path)
    if organization is None:
        organization = assist.make_organization()
    data_loader = assist.make_data_loader(dataset)
    metric = Metric({'train': ['Loss', 'Loss_Local'], 'test': ['Loss']})
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        logger.safe(True)
        train(data_loader, assist, organization, metric, logger, epoch)
        organization_outputs = broadcast(data_loader, organization, epoch)
        assist.update(epoch - 1, data_loader, organization_outputs)
        test(data_loader, assist, organization, metric, logger, epoch)
        logger.safe(False)
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'assist': assist, 'organization': organization, 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)
    return


def train(data_loader, assist, organization, metric, logger, epoch):
    start_time = time.time()
    num_active_users = len(organization)
    for i in range(num_active_users):
        organization[i].train(epoch - 1, data_loader[i]['train'], metric, logger,
                              assist.organization_outputs[i]['train'])
        if i % int((num_active_users * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * local_time * num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / num_active_users),
                             'ID: {}/{}'.format(i + 1, num_active_users),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    return


def test(data_loader, assist, organization, metric, logger, epoch):
    with torch.no_grad():
        num_active_users = len(organization)
        for i in range(num_active_users):
            organization[i].test(epoch - 1, data_loader[i]['test'], metric, logger,
                                 assist.organization_outputs[i]['test'])
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return


def broadcast(data_loader, organization, epoch):
    with torch.no_grad():
        num_active_users = len(organization)
        organization_outputs = [{split: None for split in cfg['data_size']} for _ in range(num_active_users)]
        for i in range(num_active_users):
            for split in organization_outputs[i]:
                organization_outputs[i][split] = organization[i].broadcast(epoch - 1, data_loader[i][split])
    return organization_outputs


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
        feature_split = split_dataset(cfg['num_users'], cfg['feature_split_mode'])
        assist = Assist(feature_split, cfg['assist_rate'])
        organization = None
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
    return last_epoch, assist, organization, logger


if __name__ == "__main__":
    main()