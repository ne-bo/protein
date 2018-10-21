import torch

from base import BaseModel
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50

from model.modules.spoc_layer import Spoc, create_initial_pca_matrix_for_dataset


class NatashaProtein(BaseModel):
    def __init__(self, config):
        super(NatashaProtein, self).__init__(config)
        self.channel_to_3 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        self.net = resnet34(pretrained=True)
        self.net.cnv1 = nn.Conv2d(in_channels=4, out_channels=64,kernel_size=7,stride=2, padding=3, bias=False)
        num_ftrs = self.net.fc.in_features
        self.net.fc = nn.Linear(num_ftrs, config['class_number'])
        # self.classifier = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.channel_to_3(x)
        output = self.net(x)
        #output = self.classifier(output)
        return output


class RetailModel(BaseModel):
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
        super(RetailModel, self).__init__(config)
        self.config = config
        self.desired_dimension = self.config['model']['desired_embedding_dimension']

        basic_net = resnet50(pretrained=True)
        self.net.cnv1 = nn.Conv2d(in_channels=4, out_channels=64,kernel_size=7,stride=2, padding=3, bias=False)

        self.representation_network = nn.Sequential(*list(basic_net.children())[:8])
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
        return x