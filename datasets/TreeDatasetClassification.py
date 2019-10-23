import torch

from .TreeDataset import TreeDataset


class TreeDatasetClassification(TreeDataset):

    def __init__(self, data_dir, batch_size=1, shuffle=False, word_embeddings=None):
        super().__init__(data_dir, batch_size=batch_size, shuffle=shuffle, word_embeddings=word_embeddings)
        self.n_classes = max(self.labels) + 1

    def read_labels(self, label_path):
        with open(label_path) as f:
            labels = f.readlines()
        labels = [int(l.strip()) for l in labels]
        return labels

    def label_tensor(self, label):
        return torch.LongTensor([label])

    def __getitem__(self, item):
        tree_tensor, label = super().__getitem__(item)
        label = label.squeeze(1)
        return tree_tensor, label
