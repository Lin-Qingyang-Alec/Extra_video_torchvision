# Extra_video_torchvision
Some extra video transforms in pytorch.

The idea was to increase some extra [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/) transforms for video inputs. (The code is therefore widely based on the code from this repository)

This repository works with torchvision well. It means that the Transforms can be composed just as in torchvision with video_transforms.Compose.

The input is required as the tensor with size [C, T, H, W].

## Transforms ##
### Extra ###
- GrayscaleVideo
- ColorJitterVideo
- ResizeVideo
- RandomErasingVideo
- AddSaltPepperNoiseVideo

### Original ###
- RandomSampleClip
- InterleavedSampling

RandomSampleClip and InterleavedSampling is my original transforms which I think there are simple but useful. 
RandomSampleClip randomly sample a fixed length subclip in the video. 
InterleavedSampling sampling video at intervals which means that it can sample one frame in several frames.

You can also find

- RandomCropVideo
- RandomResizedCropVideo
- CenterCropVideo
- NormalizeVideo
- ToTensorVideo
- RandomHorizontalFlipVideo

in **torchvision.transforms._transforms_video**

## How to use ##
1. Copy the path src to your project
2. Import the transforms from src/tranforms_video
3. Enjoy!!



