import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from i3d import InceptionI3d


class SyncI3d(nn.Module):
    def __init__(self, num_in_frames=64, in_features=2048, nb_classes=2, nb_layers=3, dropout_prob=0):
        super(SyncI3d, self).__init__()
        self.params_file = "/home/adrien/Code/human_interaction_SyncI3d/params/rgb_imagenet.pt"
         
        self.i3d_net = InceptionI3d(num_in_frames=num_in_frames)
        self.i3d_net.load_state_dict(torch.load(self.params_file))

        self.dropout = nn.Dropout(p=dropout_prob)

        self.in_features = in_features
        self.nb_classes = nb_classes
        self.nb_layers = nb_layers
        feature_sizes = [self.in_features // 2**i for i in range(self.nb_layers)] + [self.nb_classes]
        self.layers = nn.ModuleList([nn.Linear(feature_sizes[i], feature_sizes[i + 1]) for i in range(self.nb_layers)])
    
    def forward(self, input1, input2):
        features1 = self.i3d_net.extract_features(input1)
        features2 = self.i3d_net.extract_features(input2)

        features1 = torch.flatten(features1, start_dim=1)
        features2 = torch.flatten(features2, start_dim=1)

        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)

        x = torch.cat((features1, features2), dim=1)
        for layer in self.layers[:-1]:
            x = self.dropout(x)
            x = F.relu(layer(x))
        x = self.layers[-1](x)

        return x


class SyncI3dResNet(nn.Module):
    def __init__(self, num_in_frames=64, in_features=2048*2, nb_classes=2, nb_layers=3, dropout_prob=0):
        super(SyncI3dResNet, self).__init__()
        self.params_file = "/home/adrien/Code/human_interaction_SyncI3d/params/rgb_imagenet.pt"
         
        self.i3d_net = InceptionI3d(num_in_frames=num_in_frames)
        self.i3d_net.load_state_dict(torch.load(self.params_file))

        self.init_resnet()

        self.dropout = nn.Dropout(p=dropout_prob)

        self.in_features = in_features
        self.nb_classes = nb_classes
        self.nb_layers = nb_layers
        feature_sizes = [self.in_features // 2**i for i in range(self.nb_layers)] + [self.nb_classes]
        self.layers = nn.ModuleList([nn.Linear(feature_sizes[i], feature_sizes[i + 1]) for i in range(self.nb_layers)])
    
    def init_resnet(self):
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(2048, 2048, bias=False)
        self.resnet.fc.weight = torch.nn.Parameter(torch.eye(2048))
        self.resnet.fc.requires_grad_(False)

    def forward(self, input1, input2, input3):
        features1 = self.i3d_net.extract_features(input1)
        features2 = self.i3d_net.extract_features(input2)
        features3 = self.resnet(input3)

        features1 = torch.flatten(features1, start_dim=1)
        features2 = torch.flatten(features2, start_dim=1)
        features3 = torch.flatten(features3, start_dim=1)

        features1 = F.normalize(features1, p=2, dim=1)
        features2 = F.normalize(features2, p=2, dim=1)
        features3 = F.normalize(features3, p=2, dim=1)

        x = torch.cat((features1, features2, features3), dim=1)
        for layer in self.layers[:-1]:
            x = self.dropout(x)
            x = F.relu(layer(x))
        x = self.layers[-1](x)

        return x


if __name__ == "__main__":
    # model = SyncI3d(num_in_frames=16)
    # model.cuda()

    # torch.manual_seed(0)
    # input1 = torch.rand(4, 3, 16, 224, 224).cuda()
    # input2 = torch.rand(4, 3, 16, 224, 224).cuda()

    # output = model(input1, input2)
    # print(output.shape)

    model = SyncI3dResNet(num_in_frames=16)
    model.cuda()

    torch.manual_seed(0)
    input1 = torch.rand(4, 3, 16, 224, 224).cuda()
    input2 = torch.rand(4, 3, 16, 224, 224).cuda()
    input3 = torch.rand(4, 3, 224, 224).cuda()

    output = model(input1, input2, input3)
    print(output.shape)

