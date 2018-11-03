import sys

sys.path.append('/home/ubuntu/anaconda3/lib/python3.6/site-packages/faiss//python/')
sys.path.append('/home/ubuntu/anaconda3/lib/python3.6/site-packages/faiss/')

import faiss
import numpy as np

# Be careful in case you queries = index_set:
# This function returns the query itself as the nearest neighbor
from tqdm import tqdm

from data_loader import ProteinDataLoader
from inference import outputs_for_large_dataset, read_inference_results_from_disk
from utils.util import get_all_test_ids, get_all_train_ids, write_predictions_to_the_submission_file
from model.model import RetrievalModel


def get_nearest_neighbors(queries, index_set, k):
    dimension = queries.shape[1]
    queries = queries.cpu().numpy().astype('float32')
    index_set = index_set.cpu().numpy().astype('float32')

    resource = faiss.StandardGpuResources()

    queries_number = queries.shape[0]
    index_to_test = np.random.randint(low=0, high=queries_number)
    print('queries[index_to_test] ', queries[index_to_test])
    assert np.abs(np.linalg.norm(queries[index_to_test]) - 1.0) <= 10e-1, \
        'Cosine similarity nearest neighbors search should work with normalized data! ' \
        'But np.linalg.norm(queries[%d] = %.10f' % (index_to_test, np.linalg.norm(queries[index_to_test]))
    index = faiss.GpuIndexFlatIP(resource, dimension)

    index.add(index_set)
    s, i = index.search(queries, k)

    return s, i


def inference_for_knn(config):
    train_loader = ProteinDataLoader(config, name='train')
    test_loader = ProteinDataLoader(config, name='test')
    model = RetrievalModel(config, train_loader).eval()

    # all_outputs_train = outputs_for_large_dataset(train_loader, model, pack_volume=10, cpu=False)
    # all_outputs_test = outputs_for_large_dataset(test_loader, model, pack_volume=5851, cpu=False)

    all_outputs_train = read_inference_results_from_disk(config, batches_number=25, name='train', pack_volume=10)
    all_outputs_test = read_inference_results_from_disk(config, batches_number=1, name='test', pack_volume=5851)

    all_test_ids = get_all_test_ids(config)
    all_train_ids = get_all_train_ids(config)

    print('all_outputs_train ', all_outputs_train, all_outputs_train.shape)
    print('all_outputs_test ', all_outputs_test, all_outputs_test.shape)
    s, i = get_nearest_neighbors(queries=all_outputs_test, index_set=all_outputs_train, k=2)

    # get predictions
    predictions = []
    print('train_loader.dataset.targets ', train_loader.dataset.labels)
    for id, index_of_the_nearest in tqdm(zip(all_test_ids, i)):
        print('index_of_the_nearest ', index_of_the_nearest)
        prediction = []
        print('index_of_the_nearest ', index_of_the_nearest)
        for neighbor in index_of_the_nearest:
            print('neighbor ', neighbor, ' train_loader.dataset.labels[neighbor] ',
                  train_loader.dataset.labels[neighbor])
            prediction.extend(train_loader.dataset.labels[neighbor])
        prediction = list(set(prediction))
        print('id ', id, 'prediction', prediction, end='')
        prediction = str(prediction).replace(',', '').replace('[', '').replace(']', '')

        predictions.append(prediction)

    write_predictions_to_the_submission_file(all_test_ids, predictions)
