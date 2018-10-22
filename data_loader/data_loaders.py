import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SequentialSampler

from base import BaseDataLoader
from data_loader.sampling import UniformSampler
from datasets import protein_channels
import numpy as np


class ProteinDataLoader(DataLoader):
    """
    Protein data loading
    """

    def __init__(self, config, name, shuffle=False):
        super(ProteinDataLoader, self).__init__(
            dataset=protein_channels.ProteinChannelsDataset(config=config, name=name),
            batch_size=config['data_loader']['batch_size_%s' % name],
            drop_last=config['data_loader']['drop_last'],
            shuffle=shuffle
        )
        if False:#config['sampling'] == 'uniform' and name == 'train':
            number_of_different_classes_in_batch = 2
            batches_number = self.dataset.__len__() * number_of_different_classes_in_batch // self.batch_size

            uniform_sampler = UniformSampler(
                data_source=self.dataset,
                batches_number=batches_number,
                number_of_different_classes_in_batch=number_of_different_classes_in_batch,
                batch_size=self.batch_size
            )
            self.batch_sampler = BatchSampler(
                uniform_sampler,
                batch_size=self.batch_size,
                drop_last=self.drop_last
            )
        else:
            self.batch_sampler = BatchSampler(
                SequentialSampler(self.dataset),
                batch_size=self.batch_size,
                drop_last=self.drop_last
            )

        self.config = config
