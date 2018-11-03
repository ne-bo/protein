import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn as nn

from inference import outputs_for_large_dataset, read_inference_results_from_disk, save_inference_results_on_disk
from model.modules.l2_normalization import L2Normalization


# The paper about SPoC descriptors
# Artem Babenko, Victor Lempitsky.
#         \textit{Aggregating Deep Convolutional Features for Image Retrieval}.
#         2015
# https://arxiv.org/pdf/1510.07493.pdf

def learn_pca_matrix_for_spocs_with_sklearn(spocs, desired_dimension):
    pca = PCA(n_components=desired_dimension)
    u, s, v = pca._fit(torch.t(spocs).cpu().numpy())
    return u[:, :desired_dimension], s[:desired_dimension]


def create_initial_pca_matrix_for_dataset(train_loader, desired_dimension, network):
    config = train_loader.config
    if 'initial_PCA_matrix.npy' not in os.listdir(config['temp_folder']):
        pca_matrix, singular_values = create_new_pca_matrix_for_dataset(desired_dimension, network, train_loader)
    else:
        pca_matrix, singular_values = download_existing_pca_matrix(config['temp_folder'])
    return pca_matrix, singular_values


def download_existing_pca_matrix(temp_folder):
    pca_matrix_filename = os.path.join(temp_folder, 'initial_PCA_matrix.npy')
    print('Downloading initial PCA matrix from %s' % pca_matrix_filename)
    pca_matrix = np.load(pca_matrix_filename)
    singular_values = np.load(os.path.join(temp_folder, 'initial_singular_values.npy'))
    return pca_matrix, singular_values


def create_new_pca_matrix_for_dataset(desired_dimension: int, network, train_loader):
    config = train_loader.config
    print('Creating initial PCA matrix')
    spoc_before_pca = SpocBeforeDimensionReduction()

    batches_number = save_inference_results_on_disk(train_loader, network, name='train', pack_volume=10)

    # This is code for super large dataset to create a pca matrix using only part of all batches
    # after the inference is already done and stored on the disk
    # here we just read as many batches as we can
    path = os.path.join(config['temp_folder'], 'train', '')
    all_outputs = read_inference_results_from_disk(config, batches_number=10, name='train', pack_volume=10)

    all_outputs = spoc_before_pca(all_outputs)
    pca_matrix, singular_values = learn_pca_matrix_for_spocs_with_sklearn(all_outputs.data, desired_dimension)
    np.save(os.path.join(config['temp_folder'],
                         'initial_PCA_matrix'), pca_matrix)
    np.save(os.path.join(config['temp_folder'],
                         'initial_singular_values'), singular_values)
    return pca_matrix, singular_values


class SpocBeforeDimensionReduction(nn.Module):
    def __init__(self):
        super(SpocBeforeDimensionReduction, self).__init__()
        # L2 - normalization
        self.normalization_before_dimension_reduction = L2Normalization()

    def forward(self, inp):
        batch_size = inp.size(0)
        dimension = inp.size(1)
        # sum pooling
        spocs = torch.sum(inp.view(batch_size, dimension, -1), dim=2)
        # normalization
        spocs = self.normalization_before_dimension_reduction(spocs)
        return spocs

    def __repr__(self):
        return self.__class__.__name__


class Spoc(nn.Module):
    def __init__(self, initial_pca_matrix, initial_singular_values):
        super(Spoc, self).__init__()
        self.spoc_before_dimension_reduction = SpocBeforeDimensionReduction()
        # PCA
        self.pca_matrix = nn.Parameter(initial_pca_matrix, requires_grad=True)
        self.singular_values = nn.Parameter(initial_singular_values, requires_grad=True)
        # L2 - normalization
        self.normalization_after_dimension_reduction = L2Normalization()

    def forward(self, inp):
        # before PCA
        spocs = self.spoc_before_dimension_reduction(inp)
        # PCA
        spocs = torch.div(torch.mm(spocs, self.pca_matrix), self.singular_values)
        # normalization
        # we need this normalization step in order to use representation-specific losses
        spocs = self.normalization_after_dimension_reduction(spocs)

        return spocs

    def __repr__(self):
        return self.__class__.__name__
