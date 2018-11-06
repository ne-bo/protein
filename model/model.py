import torch
import numpy as np
from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50, resnet152, densenet121, vgg16

from model.modules.spoc_layer import Spoc, create_initial_pca_matrix_for_dataset


class NatashaProtein(BaseModel):
    def __init__(self, config):
        super(NatashaProtein, self).__init__(config)
        self.channel_to_3 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)

        # self.net = resnet50(pretrained=True)
        # num_ftrs = self.net.fc.in_features
        # self.net.fc = nn.Linear(num_ftrs, config['class_number'])

        # self.net = densenet121(pretrained=True)
        # num_ftrs = self.net.classifier.in_features
        # self.net.classifier = nn.Linear(in_features=num_ftrs, out_features=config['class_number'])

        self.net = vgg16(pretrained=True)
        self.net.classifier.add_module(module=nn.Linear(in_features=1000, out_features=config['class_number']))
        print(self.net)

    def forward(self, x):
        x = self.channel_to_3(x)
        # print('x after channel_to_3 ', x.shape)
        output = self.net(x)

        return output


class RetrievalModel(BaseModel):
    def __init__(self, config, loader_for_pca_initialization):
        """
        here we cut off this tail of Resnet in order to replace it with the Spoc layer
            (avgpool): AvgPool2d(kernel_size=7, stride=1, padding=0)
            (fc): Linear(in_features=2048, out_features=1000, bias=True)
        Actually it's not ideally clear which layer is better to take,
        but we should start somewhere

        I took the idea about Resnet from this article, but combined it with SPoCs
        Albert Gordo, Jon Almazan, Jerome Revaud, Diane Larlus.
        \textit{Deep Image Retrieval: Learning global representations for image search}.
        2016
        """
        super(RetrievalModel, self).__init__(config)
        self.config = config
        self.desired_dimension = self.config['model']['desired_embedding_dimension']

        self.channel_to_3 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        basic_net = resnet50(pretrained=True)

        self.representation_network = nn.Sequential(self.channel_to_3, *list(basic_net.children())[:8])
        self.train_loader = loader_for_pca_initialization
        initial_pca_matrix, initial_singular_values = create_initial_pca_matrix_for_dataset(
            loader_for_pca_initialization,
            self.desired_dimension,
            self.representation_network
        )
        self.spoc_layer = Spoc(
            torch.from_numpy(initial_pca_matrix).cuda(torch.device('cuda:' + str(config['gpu']))),
            torch.from_numpy(initial_singular_values).cuda(torch.device('cuda:' + str(config['gpu'])))
        )

    def forward(self, x):
        x = self.representation_network(x)
        x = self.spoc_layer(x)

        index_to_test = np.random.randint(low=0, high=x.shape[0])
        # print('x[index_to_test] ', x[index_to_test])
        # assert np.abs(np.linalg.norm(x[index_to_test].detach().cpu().numpy()) - 1.0) <= 10e-1, \
        #     'Spoc should contain normalized data! ' \
        #     'But np.linalg.norm(x[%d] = %.10e' % (index_to_test, np.linalg.norm(x[index_to_test].detach().cpu().numpy()))
        return x
