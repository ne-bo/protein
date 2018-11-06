import os
import jstyleson
import logging
import argparse
import torch

from inference import inference
from knn.knn_classification import inference_for_knn

from model.model import NatashaProtein, RetrievalModel
from model.loss import *
from model.metric import *
from data_loader import ProteinDataLoader, ProteinDataLoader
from trainer import Trainer
from logger import Logger
from trainer.similarity_trainer import SimilarityTrainer

logging.basicConfig(level=logging.INFO, format='')


def main(config, resume):
    train_logger = Logger()

    data_loader = ProteinDataLoader(config, name='train', shuffle=True, evaluation=False)
    valid_data_loader = None  # data_loader.split_validation()

    if False:  # config['similarity_approach']:
        assert config['sampling'] == 'uniform'
        assert config['loss'] == 'histogram_loss'
        model = RetrievalModel(config, data_loader).cuda()
    else:
        model = NatashaProtein(config=config).cuda()

    loss = eval(config['loss'])
    metrics = [eval(metric) for metric in config['metrics']]

    if True:
        print('start training')
        if False:  # config['similarity_approach']:
            trainer = SimilarityTrainer(model, loss, metrics,
                                        resume=resume,
                                        config=config,
                                        data_loader=data_loader,
                                        valid_data_loader=valid_data_loader,
                                        train_logger=train_logger)
        else:
            trainer = Trainer(model, loss, metrics,
                              resume=resume,
                              config=config,
                              data_loader=data_loader,
                              valid_data_loader=valid_data_loader,
                              train_logger=train_logger)
        print('batch_size ', config['data_loader']['batch_size_train'])
        trainer.train()

    print('Create test loader')
    test_data_loader = ProteinDataLoader(config, name='test')
    checkpoint_for_model = torch.load('saved/Protein/model_best.pth.tar')
    # checkpoint_for_model = torch.load('saved/Protein/model_best-resnent-50-lb-0.383.pth.tar')
    # checkpoint_for_model = torch.load('saved/NatashaSegmentation/model_best-resnet-34-wo-depth-bce-0.0063.pth.tar')
    model.load_state_dict(checkpoint_for_model['state_dict'])
    # print(model.state_dict())
    model.eval()

    print('Do inference')

    if False:  # config['similarity_approach']:
        inference_for_knn(config)
    else:
        inference(test_data_loader, model)


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.resume is not None:
        if args.config is not None:
            logger.warning('Warning: --config overridden by --resume')
        config = torch.load(args.resume)['config']
    elif args.config is not None:
        config = jstyleson.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        # assert not os.path.exists(path), "Path {} already exists!".format(path)
    assert config is not None

    main(config, args.resume)
