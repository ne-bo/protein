import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def outputs_for_large_dataset(loader, network):
    config = loader.config
    torch.cuda.empty_cache()
    name = loader.dataset.name
    batches_number = save_inference_results_on_disk(loader, network, name)
    # batches_number = 9
    name = 'test'
    return read_inference_results_from_disk(config, batches_number, name)


def read_inference_results_from_disk(config, batches_number, name):
    path = os.path.join(config['temp_folder'], name, '')
    pack_volume = config['pack_volume']
    assert 'all_outputs_%d' % pack_volume in os.listdir(path), \
        'There should be precomputed inference data in %s!' % path

    all_outputs = torch.cuda.FloatTensor()
    for i in range(1, batches_number + 1):
        outputs = torch.load('%sall_outputs_%d' % (path, i * pack_volume))
        all_outputs = torch.cat((all_outputs, outputs), dim=0)

    return all_outputs


def save_inference_results_on_disk(loader, network, name):
    config = loader.config
    pack_volume = config['pack_volume']
    path = os.path.join(config['temp_folder'], name, '')
    print('path ', path)
    network.eval()
    network = network.cuda()
    all_outputs = torch.cuda.FloatTensor()
    i = 1
    print('Inference is in progress')
    for data in tqdm(loader):
        images, targets = data

        images = images.cuda()

        outputs = network(images).detach()
        outputs = torch.nn.Sigmoid()(outputs)

        all_outputs = torch.cat((all_outputs, outputs.data), dim=0)

        if i % pack_volume == 0:
            torch.save(all_outputs, '%sall_outputs_%d' % (path, i))
            all_outputs = torch.cuda.FloatTensor()
            torch.cuda.empty_cache()
        i += 1
    batches_number = len(loader) // pack_volume
    print('batches_number = ', batches_number)
    all_outputs = None
    torch.cuda.empty_cache()
    return batches_number


def inference(loader, model):
    config = loader.config

    all_outputs = outputs_for_large_dataset(loader, model)

    all_ids = []
    sample_submission = os.path.join(config['data_loader']['data_dir'], 'sample_submission.csv')
    with open(sample_submission, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',', dialect='excel')
        rows = list(reader)
        for i, row in tqdm(enumerate(rows[1:])):
            all_ids.append(row[0])

    predictions = []
    for id, output in tqdm(zip(all_ids, all_outputs)):
        print('id ', id, end='')
        predictions.append(convert_output_to_prediction(output))

    rows = []
    with open('natasha_submission.csv', 'w') as csv_file:
        csv_file.write('Id,Predicted\n')
        for (id, prediction) in tqdm(zip(all_ids, predictions)):
            row = str(id) + ',' + prediction + '\n'
            rows.append(row)
        rows[-1] = rows[-1].replace('\n', '')
        csv_file.writelines(rows)


def convert_output_to_prediction(output):
    prediction = ''
    output = output.cpu().numpy()
    indices = np.argsort(output)[::-1][:5]
    print('output(indices)', output[indices])
    for i, class_number in enumerate(indices):
        if output[class_number] > (1.0/28.0)*5.0 or i==0:
            prediction = prediction + ' ' + str(class_number)
    prediction = prediction[1:]
    return prediction
