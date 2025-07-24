import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Sampler
import random


class SimpleDataset(Dataset):
    def __init__(self, data_content):
        self.data = data_content['data']
        self.class_label = data_content['label']

    def __getitem__(self, index):
        data_i = self.data[index]
        class_label_i = self.class_label[index]

        return data_i, class_label_i

    def __len__(self):
        if len(self.data) != len(self.class_label):
            raise ValueError('The number of samples should be equal to the number of labels!')
        return len(self.data)

    def get_class_labels(self):
        return self.class_label


class BalancedBatchSampler(Sampler):
    def __init__(self, cls_labels, batch_size):
        self.cls_labels = cls_labels
        self.batch_size = batch_size

        self.label_indices = {}

        for idx, cls_label in enumerate(cls_labels):
            if cls_label not in self.label_indices:
                self.label_indices[cls_label] = []
            self.label_indices[cls_label].append(idx)

        self.samples_per_label = batch_size // len(self.label_indices)

    def __iter__(self):
        all_indices = []
        for label in self.label_indices.values():
            all_indices.extend(label)
        random.shuffle(all_indices)

        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        return self.samples_per_label * len(self.label_indices)
