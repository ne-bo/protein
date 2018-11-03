import torch

import numpy as np
from torch.utils.data.sampler import Sampler


class UniformSampler(Sampler):
    """Samples elements with roughly uniform distribution of samples with the same label
    Arguments:
    """

    def __init__(self, data_source,
                 batch_size,
                 number_of_different_classes_in_batch,
                 batches_number):
        """

        :param data_source: dataset, should be an inheritor of Dataset
        :param batch_size: desired batch size, int
        :param number_of_different_classes_in_batch: desired number of different classes in batch,usually 2 or 3
        :param batches_number: how many batches you want to create
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.labels = self.data_source.labels
        self.length = len(self.labels)  # how many samples we have in our dataset
        self.number_of_samples_with_the_same_label_in_the_batch = batch_size // number_of_different_classes_in_batch
        self.number_of_different_classes_in_batch = number_of_different_classes_in_batch
        self.batches_number = batches_number

    def draw_samples_with_label(self, label):
        labels_array = np.array(self.labels)
        # print('labels_array ', labels_array)
        # selected_samples = np.where(labels_array.any() == label)[0]
        selected_samples = []
        for i, labels_list in enumerate(labels_array):
            # print('label', label, 'labels_list ', labels_list)
            if label in labels_list:
                selected_samples.append(i)
        selected_samples = np.array(selected_samples)
        # print('selected_samples ', selected_samples, len(selected_samples))
        # print('self.number_of_samples_with_the_same_label_in_the_batch ', self.number_of_samples_with_the_same_label_in_the_batch)
        # in case we have too few sample with this label we just duplicate examples
        total_selected_samples = selected_samples
        while total_selected_samples.shape[0] < self.number_of_samples_with_the_same_label_in_the_batch:
            total_selected_samples = np.hstack((total_selected_samples, selected_samples))

        # shuffle
        samples = np.random.permutation(total_selected_samples)
        # take the requested number of samples
        samples = samples[:self.number_of_samples_with_the_same_label_in_the_batch]
        #print('samples ', samples)
        return samples

    def get_new_batch(self):
        batch = np.array([], dtype=int)
        labels_already_in_batch = []
        for class_number in range(self.number_of_different_classes_in_batch):
            label = np.random.choice(np.array(np.random.choice(np.array(self.labels))))
            # print('our random label is ', label)
            #print('labels_already_in_batch ', labels_already_in_batch)
            while label in labels_already_in_batch:
                label = np.random.choice(np.array(np.random.choice(np.array(self.labels))))
            labels_already_in_batch.append(label)
            batch = np.hstack((batch, self.draw_samples_with_label(label)))
        #print('batch ', batch)
        return batch

    def __iter__(self):
        batches = np.array([], dtype=int)
        for batch_number in range(self.batches_number):
            new_batch = self.get_new_batch()
            batches = np.hstack((batches, new_batch))
        return iter(np.array(batches))

    def __len__(self):
        return self.batches_number * \
               self.number_of_different_classes_in_batch * \
               self.number_of_samples_with_the_same_label_in_the_batch
