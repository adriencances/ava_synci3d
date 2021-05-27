import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from i3d import InceptionI3d


class SyncI3d_v1(nn.Module):
    def __init__(self, num_in_frames=64, in_features=2048, nb_classes=2, nb_layers=3):
        super(SyncI3d_v1, self).__init__()
        self.params_file = "/home/acances/Code/human_interaction_SyncI3d/params/rgb_imagenet.pt"
         
        self.i3d_net = InceptionI3d(num_in_frames=num_in_frames)
        self.i3d_net.load_state_dict(torch.load(self.params_file))

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
            x = F.relu(layer(x))
        x = self.layers[-1](x)

        return x


if __name__ == "__main__":
    model = SyncI3d_v1(num_in_frames=16)
    model.cuda()

    torch.manual_seed(0)
    input1 = torch.rand(4, 3, 16, 224, 224).cuda()
    input2 = torch.rand(4, 3, 16, 224, 224).cuda()

    output = model(input1, input2)
    print(output.shape)

