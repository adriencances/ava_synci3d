import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from i3d import InceptionI3d


class SyncI3d(nn.Module):
    def __init__(self):
        super(SyncI3d, self).__init__()
        self.params_file = "/home/acances/Code/human_interaction_SyncI3d/params/rgb_imagenet.pt"
         
        self.i3d_net_1 = InceptionI3d(num_in_frames=64)
        self.i3d_net_1.load_state_dict(torch.load(self.params_file))

        self.i3d_net_2 = InceptionI3d(num_in_frames=64)
        self.i3d_net_2.load_state_dict(torch.load(self.params_file))
    
    def forward(self, input1, input2):
        output1 = self.i3d_net_1.extract_features(input1)
        output1 = torch.flatten(output1, start_dim=1)

        output2 = self.i3d_net_2.extract_features(input2)
        output2 = torch.flatten(output2, start_dim=1)
        
        return output1, output2


if __name__ == "__main__":
    model = SyncI3d()
    model.cuda()
    # print(model.size)

    torch.manual_seed(0)
    input1 = torch.rand(1, 3, 64, 224, 224).cuda()
    input2 = torch.rand(1, 3, 64, 224, 224).cuda()

    output1, output2 = model(input1, input2)
    print(output1.shape)
    print(output2.shape)

