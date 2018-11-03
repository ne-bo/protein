import os

import numpy as np
import torch
from tqdm import tqdm

from utils.util import write_predictions_to_the_submission_file, get_all_test_ids


def outputs_for_large_dataset(loader, network, pack_volume, cpu=False):
    config = loader.config
    torch.cuda.empty_cache()
    name = loader.dataset.name
    batches_number = save_inference_results_on_disk(loader, network, name, pack_volume, cpu)
    torch.cuda.empty_cache()
    print('batches_number ', batches_number)
    return read_inference_results_from_disk(config, batches_number, name, pack_volume)


def read_inference_results_from_disk(config, batches_number, name, pack_volume, start_batch=1):
    torch.cuda.empty_cache()
    path = os.path.join(config['temp_folder'], name, '')
    if pack_volume is None:
        pack_volume = config['pack_volume']
    print('pack_volume  in read ', pack_volume)
    assert 'all_outputs_%d' % pack_volume in os.listdir(path), \
        'There should be precomputed inference data in %s!' % path

    all_outputs = torch.cuda.FloatTensor()
    for i in range(start_batch, batches_number + 1):
        print('i ', i)
        outputs = torch.load('%sall_outputs_%d' % (path, i * pack_volume))
        print('%sall_outputs_%d' % (path, i * pack_volume), 'outputs ', outputs)
        all_outputs = torch.cat((all_outputs, outputs), dim=0)
        outputs = None
        torch.cuda.empty_cache()

    return all_outputs


def save_inference_results_on_disk(loader, network, name, pack_volume=None, cpu=False):
    config = loader.config
    if pack_volume is None:
        pack_volume = config['pack_volume']
    print('pack_volume  in save ', pack_volume)
    path = os.path.join(config['temp_folder'], name, '')
    print('path ', path)
    network.eval()
    if not cpu:
        network = network.cuda()
        all_outputs = torch.cuda.FloatTensor()
    else:
        all_outputs = torch.FloatTensor()
    i = 1
    print('Inference is in progress')
    print('len(loader) ', len(loader))
    for data in tqdm(loader):
        images, targets = data

        if not cpu:
            images = images.cuda()

        # outputs = network(images).detach()

        outputs1 = network(images[:, :, :224, :224]).detach()
        outputs2 = network(images[:, :, 288:, :224]).detach()
        outputs3 = network(images[:, :, :224, 288:]).detach()
        outputs4 = network(images[:, :, 288:, 288:]).detach()
        outputs5 = network(images[:, :, 144:368, 144:368]).detach()
        outputs = (outputs1 + outputs2 + outputs3 + outputs4 + outputs5) / 5.0

        if not config['similarity_approach']:
            outputs = torch.nn.Sigmoid()(outputs)

        all_outputs = torch.cat((all_outputs, outputs.data), dim=0)

        if i % pack_volume == 0:
            torch.save(all_outputs, '%sall_outputs_%d' % (path, i))
            if not cpu:
                all_outputs = torch.cuda.FloatTensor()
            else:
                all_outputs = torch.FloatTensor()
            torch.cuda.empty_cache()
        i += 1
    print('len(loader) after ', len(loader))
    batches_number = len(loader) // pack_volume
    print('batches_number = ', batches_number)
    all_outputs = None
    torch.cuda.empty_cache()
    return batches_number


def inference(loader, model):
    config = loader.config

    all_outputs = outputs_for_large_dataset(loader, model, pack_volume=5851)
    print('all_outputs ', all_outputs.shape)
    all_ids = get_all_test_ids(config)

    predictions = []
    for id, output in tqdm(zip(all_ids, all_outputs)):
        print('id ', id, end='')
        predictions.append(convert_output_to_prediction(output))

    write_predictions_to_the_submission_file(all_ids, predictions)


def convert_output_to_prediction(output):
    prediction = ''
    output = output.cpu().numpy()
    indices = np.argsort(output)[::-1][:5]
    print('output(indices)', output[indices])
    for i, class_number in enumerate(indices):
        if output[class_number] > (1.0 / 28.0) * 2.0 or i == 0:
            prediction = prediction + ' ' + str(class_number)
    prediction = prediction[1:]
    return prediction
