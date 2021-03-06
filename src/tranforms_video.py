import math
import numbers
import numpy as np
import random
import torch
from src import functional_video as F

from torch import Tensor
from typing import Tuple, List, Optional

__all__ = ['GrayscaleVideo', 'ColorJitterVideo', 'ResizeVideo', 'RandomErasingVideo', 'AddSaltPepperNoiseVideo', 'RandomSampleClip', 'InterleavedSampling']

class GrayscaleVideo(torch.nn.Module):
    """Convert image to grayscale.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [C, T, H, W].

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
         - If ``num_output_channels == 1`` : returned image is single channel
         - If ``num_output_channels == 3`` : returned image is 3 channel with r == g == b

    """

    def __init__(self, num_output_channels=1):
        super().__init__()
        self.num_output_channels = num_output_channels

    def forward(self, clip):
        """
        Args:
            clip (PIL Image or Tensor): Image to be converted to grayscale.

        Returns:
            PIL Image or Tensor: Grayscaled image.
        """
        return F.rgb_to_grayscale_video(clip, num_output_channels=self.num_output_channels)



class ColorJitterVideo(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, clip):
        """
        Args:
            clip (Tensor): Input clip [C, T, H, W].

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        # clip = clip.transpose(0,1)
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                clip = F.adjust_brightness(clip, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                clip = F.adjust_contrast(clip, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                clip = F.adjust_saturation(clip, saturation_factor)
            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                clip = F.adjust_hue_video(clip, hue_factor)

        # clip = torch.stack(clip, dim=0).transpose(0,1)

        return clip





class ResizeVideo(torch.nn.Module):
    """Resize the input clip to the given size.
    The clip is  a torch Tensor, in which case it is expected to have [C, T, H, W] shape.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size).
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[size, ]``.
        interpolation (int, optional): Desired interpolation enum defined by `filters`_.
            Default is ``PIL.Image.BILINEAR``. If input is Tensor, only ``PIL.Image.NEAREST``, ``PIL.Image.BILINEAR``
            and ``PIL.Image.BICUBIC`` are supported.
    """

    def __init__(self, size, interpolation="bilinear"):
        super().__init__()

        if isinstance(size, tuple):
            assert len(size) == 2, "size should be tuple (height, width)"
            self.size = size
        else:
            self.size = (size, size)
        self.interpolation = interpolation

    def forward(self, clip):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        return F.resize(clip, self.size, self.interpolation)


class RandomErasingVideo(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

     Input:
        clip (torch.tensor, dtype=torch.uint8): Size is [T, H, W, C].

    Returns:
        Erased Clip.


    """

    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    def get_params(
            self, img: Tensor, scale: Tuple[float, float], ratio: Tuple[float, float],
            value: Optional[List[float]] = None
    ) -> Tuple[int, int, int, int, Tensor]:
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (tuple or list): range of proportion of erased area against input image.
            ratio (tuple or list): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_t, img_h, img_w, img_c = img.shape
        area = img_h * img_w

        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.empty(1).uniform_(ratio[0], ratio[1]).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_t, h, w, img_c], dtype=torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            i = torch.randint(0, img_h - h + 1, size=(1,)).item()
            j = torch.randint(0, img_w - w + 1, size=(1,)).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, clip):
        """
        Args:
            clip (Tensor): Tensor clip to be erased.

        Returns:
            img (Tensor): Erased Tensor image.
        """
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value, (int, float)):
                value = [self.value, ]
            elif isinstance(self.value, str):
                value = None
            elif isinstance(self.value, tuple):
                value = list(self.value)
            else:
                value = self.value

            if value is not None and not (len(value) in (1, clip.shape[-1])):
                raise ValueError(
                    "If value is a sequence, it should have either a single value or "
                    "{} (number of input channels)".format(clip.shape[-1])
                )

            x, y, h, w, v = self.get_params(clip, scale=self.scale, ratio=self.ratio, value=value)

            clip[:, x:x + h, y:y + w, :] = v
            return clip
        return clip


class AddSaltPepperNoiseVideo(object):

    def __init__(self, density=0.1, p=0.5):
        self.density = density
        self.p = p

    def __call__(self, clip):
        if torch.rand(1) < self.p:
            t, h, w, c = clip.shape
            Nd = self.density
            Sd = 1 - Nd
            mask = np.random.choice((0, 1, 2), size=(t, h, w, 1), p=[Nd / 2.0, Nd / 2.0, Sd])
            mask = torch.tensor(mask)
            mask = mask.expand(t, h, w, 3)
            clip[mask == 0] = 0
            clip[mask == 1] = 255
            return clip
        return clip


class RandomSampleClip(object):
    '''
    Randomly sample a fixed length subclip in the clip.
    Args:
        output_length: The length of the output subclip.
        time_first: If time_first is True, the input and the output size is [T, H, W, C]. Else is [C, T, H, W].

    '''
    def __init__(self, output_length, time_first = False):
        super(RandomSampleClip, self).__init__()
        self.output_length = output_length
        self.time_first = time_first

    def __call__(self,clip):
        input_length = clip.size(0) if self.time_first is True else clip.size(1)
        assert input_length >= self.output_length
        begin = random.randint(0, input_length - self.output_length)
        return clip[begin:begin + self.output_length, ...] if self.time_first is True\
            else clip[:,begin:begin + self.output_length, :, :]


class InterleavedSampling(object):
    '''
    Sampling video at intervals.
    Args:
        rate: Sample rate.
    '''
    def __init__(self, rate):
        super(InterleavedSampling, self).__init__()
        self.rate = rate

    def __call__(self, clip):
        '''
        :param clip: (Tensor): Input clip (T, H, W, C).
        :return:
        '''
        if self.rate == 1:
            return clip
        else:
            t, _, _, _ = clip.shape
            sample = list(range(0, t, self.rate))
            return clip[sample, ...]


