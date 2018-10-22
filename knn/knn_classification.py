import faiss
import numpy as np

# Be careful in case you queries = index_set:
# This function returns the query itself as the nearest neighbor
from data_loader import ProteinDataLoader
from inference import outputs_for_large_dataset, get_all_test_ids, get_all_train_ids
from model.model import RetailModel


def get_nearest_neighbors(queries, index_set, k):
    dimension = queries.shape[1]
    queries = queries.cpu().numpy().astype('float32')
    index_set = index_set.cpu().numpy().astype('float32')

    resource = faiss.StandardGpuResources()

    queries_number = queries.shape[0]
    index_to_test = np.random.randint(low=0, high=queries_number)
    assert np.abs(np.linalg.norm(queries[index_to_test]) - 1.0) <= 10e-1, \
        'Cosine similarity nearest neighbors search should work with normalized data! ' \
        'But np.linalg.norm(queries[%d] = %.10e' % (index_to_test, np.linalg.norm(queries[index_to_test]))
    index = faiss.GpuIndexFlatIP(resource, dimension)

    index.add(index_set)
    s, i = index.search(queries, k)

    return s, i


def inference_for_knn(config):
    train_loader = ProteinDataLoader(config, name='train')
    test_loader = ProteinDataLoader(config, name='test')
    model = RetailModel(config, train_loader)
    all_outputs_train = outputs_for_large_dataset(train_loader, model)
    all_outputs_test = outputs_for_large_dataset(test_loader, model)
    all_test_ids = get_all_test_ids(config)
    all_train_ids = get_all_train_ids(config)
