import argparse
import datetime
import os
import torch
import torch.backends.cudnn as cudnn
import models
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset
from metrics import Metric
from assist import Assist
from organization import Organization
from utils import save, load, to_device, process_control, process_dataset, resume, collate
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
cfg['metric_name'] = {'test': ['Loss', 'Accuracy']}


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
    last_epoch, assist, organization, logger = resume(cfg['model_tag'], load_tag='best')
    assist.reset()
    data_loader = assist.make_data_loader(dataset)
    current_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    logger_path = 'output/runs/test_{}_{}'.format(cfg['model_tag'], current_time)
    test_logger = Logger(logger_path)
    for epoch in range(1, last_epoch):
        test_logger.safe(True)
        organization_scores = broadcast(data_loader, organization, epoch)
        assist.update(epoch - 1, data_loader, organization_scores)
        test(data_loader, assist, organization, test_logger, epoch)
        test_logger.reset()
    test_logger.safe(False)
    _, _, _, train_logger = resume(cfg['model_tag'], load_tag='checkpoint')
    save_result = {'cfg': cfg, 'epoch': last_epoch, 'logger': {'train': train_logger, 'test': test_logger}}
    save(save_result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def test(data_loader, assist, organization, logger, epoch):
    with torch.no_grad():
        num_active_users = len(organization)
        for i in range(num_active_users):
            organization[i].test(epoch - 1, data_loader[i]['test'], logger, assist.organization_scores[i]['test'])
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', cfg['metric_name']['test']))
    return


def broadcast(data_loader, organization, epoch):
    with torch.no_grad():
        num_active_users = len(organization)
        organization_scores = [{split: None for split in cfg['data_size']} for _ in range(num_active_users)]
        for i in range(num_active_users):
            for split in organization_scores[i]:
                organization_scores[i][split] = organization[i].broadcast(epoch - 1, data_loader[i][split])
    return organization_scores


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