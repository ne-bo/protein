import csv
import os

from tqdm import tqdm


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def freeze_network(network):
    for p in network.parameters():
        p.requires_grad = False


def unfreeze_network(network):
    for p in network.parameters():
        p.requires_grad = True


def write_predictions_to_the_submission_file(all_ids, predictions):
    rows = []
    with open('natasha_submission.csv', 'w') as csv_file:
        csv_file.write('Id,Predicted\n')
        for (id, prediction) in tqdm(zip(all_ids, predictions)):
            row = str(id) + ',' + prediction + '\n'
            rows.append(row)
        print(rows)
        rows[-1] = rows[-1].replace('\n', '')
        csv_file.writelines(rows)


def get_all_test_ids(config):
    return get_all_ids_from_csv(config, csv_filename='sample_submission.csv')


def get_all_train_ids(config):
    return get_all_ids_from_csv(config, csv_filename='train.csv')


def get_all_ids_from_csv(config, csv_filename):
    all_ids = []
    sample_submission = os.path.join(config['data_loader']['data_dir'], csv_filename)
    with open(sample_submission, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', dialect='excel')
        rows = list(reader)
        for i, row in tqdm(enumerate(rows[1:])):
            all_ids.append(row[0])
    print('all_ids ', all_ids)
    return all_ids
