import anytree
import numpy as np
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import check_exists, save, load
from .utils import make_classes_counts, make_tree, make_flat_index


class Wine(Dataset):
    data_name = 'Wine'

    def __init__(self, root, split, subset, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.feature, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)))
        self.target = self.target[self.subset]
        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.classes_size = load(os.path.join(self.processed_folder, 'meta.pt'))
        self.classes_to_labels, self.classes_size = self.classes_to_labels[self.subset], self.classes_size[self.subset]

    def __getitem__(self, index):
        feature, target = torch.tensor(self.feature[index]), torch.tensor(self.target[index])
        input = {'feature': feature, self.subset: target}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.feature)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            raise ValueError('Not valid dataset')
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'))
        save(test_set, os.path.join(self.processed_folder, 'test.pt'))
        save(meta, os.path.join(self.processed_folder, 'meta.pt'))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nSubset: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.subset, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        data = pd.read_csv(os.path.join(self.raw_folder, 'winequality-red.csv')).to_numpy()
        split_idx = int(data.shape[0] * 0.8)
        train_feature, test_feature = data[:split_idx, :-1].astype(np.float32), data[split_idx:, :-1].astype(np.float32)
        train_label, test_label = data[:split_idx, -1], data[split_idx:, -1]
        min_label = np.min(train_label)
        train_label = (train_label - min_label).astype(np.int64)
        test_label = (test_label - min_label).astype(np.int64)
        train_target, test_target = {'label': train_label}, {'label': test_label}
        classes_to_labels = {'label': anytree.Node('U', index=[])}
        classes = list(map(str, list(range(max(train_label) + 1))))
        for c in classes:
            make_tree(classes_to_labels['label'], [c])
        classes_size = {'label': make_flat_index(classes_to_labels['label'])}
        return (train_feature, train_target), (test_feature, test_target), (classes_to_labels, classes_size)