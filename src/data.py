import torch
import datasets
import numpy as np
from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def fetch_dataset(data_name, verbose=True):
    dataset = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\')'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\')'.format(data_name))
    elif data_name in ['MNIST', 'CIFAR10']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        cfg['transform'] = {
            'train': datasets.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]),
            'test': datasets.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        }
        dataset['train'].transform = cfg['transform']['train']
        dataset['test'].transform = cfg['transform']['test']
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag):
    data_loader = {}
    for k in dataset:
        data_loader[k] = DataLoader(dataset=dataset[k], shuffle=cfg[tag]['shuffle'][k],
                                    batch_size=cfg[tag]['batch_size'][k],
                                    pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate)
    return data_loader


def split_dataset(num_users, feature_split_mode):
    if cfg['data_name'] in ['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']:
        if feature_split_mode == 'iid':
            num_features = cfg['data_shape'][0]
            feature_split = list(torch.randperm(num_features).split(num_features // num_users))
            feature_split = feature_split[:num_users - 1] + [torch.cat(feature_split[num_users - 1:])]
        else:
            raise ValueError('Not valid feature split mode')
    elif cfg['data_name'] in ['MNIST', 'CIFAR10']:
        num_features = np.prod(cfg['data_shape'][1:]).item()
        if feature_split_mode == 'iid':
            feature_split = list(torch.randperm(num_features).split(num_features // num_users))
            feature_split = feature_split[:num_users - 1] + [torch.cat(feature_split[num_users - 1:])]
        elif feature_split_mode == 'non-iid':
            idx = torch.arange(num_features).view(*cfg['data_shape'][1:])
            power = np.log2(num_users)
            n_h, n_w = int(2 ** (power // 2)), int(2 ** (power - power // 2))
            feature_split = idx.view(n_h, cfg['data_shape'][1] // n_h, n_w, cfg['data_shape'][2] // n_w) \
                .transpose(1, 2).reshape(-1, cfg['data_shape'][1] // n_h, cfg['data_shape'][2] // n_w)
            feature_split = list(feature_split.view(feature_split.size(0), -1))
        else:
            raise ValueError('Not valid feature split mode')
    else:
        raise ValueError('Not valid data name')
    return feature_split