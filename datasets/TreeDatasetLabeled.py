import os
import torch

from .TreeDataset import TreeDataset


class TreeDatasetLabeled(TreeDataset):

    def __init__(self, data_dir, batch_size=1, shuffle=False):
        super().__init__(data_dir, batch_size=batch_size, shuffle=shuffle)
        self.labels = self.read_labels(os.path.join(data_dir, 'labels.txt'))

    def __getitem__(self, item):
        tree_tensor = super().__getitem__(item)
        batch_indices = self.epoch_batches[item]
        label = [self.label_tensor(self.labels[i]) for i in batch_indices]
        label = torch.stack(label)
        return tree_tensor, label

    def read_labels(self, label_path):
        raise NotImplementedError()

    def label_tensor(self, label):
        raise NotImplementedError()
