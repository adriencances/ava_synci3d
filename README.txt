train.py is the file to launch for option 1, 2a and 3a.
train_resnet.py is the file to launch for option 2b and 3b.

Option 1:
Simple twin I3d model (with shared weights) taking as input the two videos, with CosineEmbeddingLoss at the end of the model. This loss requires a margin that is set by default at 0.3 in the code. For prediction, another margin (prediction margin) is used, whose default value is 0.6 in the code.

Option 2a:
The two feature vectors from the twin I3d model (with shared weights) are concatenated and fed to a small MLP consisting of two linear layers (with ReLU activation), at the end of which BCEWithLogitsLoss is used with 1 class.

Option 2b:
Variant of option 2a in which a particular frame (defined below) of the video is fed to a pretrained ResNet model with fixed weights. Instead of concatenating only the two feature vectors from the twin I3d model, we concatenate these two feature vectors as well as the feature vector given by the ResNet model. As a reminder, the two input videos are two synched subvideos coming from the same video extract. The particular frame fed to the ResNet is the middle frame of this video extract.

Option 3a:
The two feature vectors from the twin I3d model (with shared weights) are concatenated and fed to a single linear layer, at the end of which BCEWithLogitsLoss is used with 1 class.

Option 3b:
Variant of option 3a which corresponds to what option 2b is to option 2a.


USAGE

Use the parser arguments to define the given options and parameters.

For help, type:

python train.py --help
OR
python train_resnet.py --help



