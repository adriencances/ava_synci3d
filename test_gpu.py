import torch
from synci3d import SyncI3d


print(torch.cuda.is_available())
print(torch.cuda.device_count())

model = SyncI3d(num_in_frames=16)
model.cuda()

input1 = torch.rand(8, 3, 16, 224, 224)
input2 = torch.rand(8, 3, 16, 224, 224)

input1 = input1.cuda()
input2 = input2.cuda()

output1, output2 = model(input1, input2)


print(output1.shape)

