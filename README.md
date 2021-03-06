# extra_video_torchvision
Some extra video transforms in pytorch.

The idea was to increase some extra torchvision transforms for video inputs. (The code is therefore widely based on the code from this repository)
This repository works with torchvision well. It means that the Transforms can be composed just as in torchvision with video_transforms.Compose.
The input is required as the tensor with size [C, T, H, W].
