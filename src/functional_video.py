import torch
from torch import Tensor
from torchvision.transforms.functional_tensor import _is_tensor_a_torch_image, convert_image_dtype

def resize(clip, target_size, interpolation_mode):
    assert len(target_size) == 2, "target size should be tuple (height, width)"
    return torch.nn.functional.interpolate(
        clip, size=target_size, mode=interpolation_mode
    )

def _rgb2hsv_video(clip):
    r, g, b = clip.unbind(dim=-4)
    #  (C, T, H, W)
    # Implementation is based on https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/
    # src/libImaging/Convert.c#L330
    maxc = torch.max(clip, dim=-4).values
    minc = torch.min(clip, dim=-4).values

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occuring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    cr = maxc - minc
    # Since `eqc => cr = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = cr / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    cr_divisor = torch.where(eqc, ones, cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor

    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc), dim=-4)


def _hsv2rgb_video(clip):
    h, s, v = clip.unbind(dim=-4)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)

    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6

    mask = i.unsqueeze(dim=-3) == torch.arange(6, device=i.device).view(-1, 1, 1)
    a1 = torch.stack((v, q, p, p, t, v), dim=-3)
    a2 = torch.stack((t, v, v, q, p, p), dim=-3)
    a3 = torch.stack((p, p, t, v, v, q), dim=-3)
    a4 = torch.stack((a1, a2, a3), dim=-4)

    result = torch.einsum("...ijk, ...xijk -> ...xjk", mask.to(dtype=clip.dtype), a4)
    return result.transpose(0,1)

def rgb_to_grayscale_video(clip: Tensor, num_output_channels: int = 1) -> Tensor:
    """
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140

    Args:
        clip (Tensor): Image to be converted to Grayscale in the form [C, T, H, W].
        num_output_channels (int): number of channels of the output image. Value can be 1 or 3. Default, 1.

    Returns:
        Tensor: Grayscale version of the image.
            if num_output_channels = 1 : returned image is single channel

            if num_output_channels = 3 : returned image is 3 channel with r = g = b

    """


    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')


    r, g, b = clip.unbind(dim=-4)
    # This implementation closely follows the TF one:
    # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(clip.dtype)
    l_img = l_img.unsqueeze(dim=-4)

    if num_output_channels == 3:
        return l_img.expand(clip.shape)

    return l_img


def _blend(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    bound = 1.0 if img1.is_floating_point() else 255.0
    return (ratio * img1 + (1.0 - ratio) * img2).clamp(0, bound).to(img1.dtype)


def adjust_brightness(clip: Tensor, brightness_factor: float) -> Tensor:
    """
    Args:
        clip (Tensor): Clip to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.

    Returns:
        Tensor: Brightness adjusted clip.
    """
    if brightness_factor < 0:
        raise ValueError('brightness_factor ({}) is not non-negative.'.format(brightness_factor))

    if not _is_tensor_a_torch_image(clip):
        raise TypeError('tensor is not a torch image.')

    return _blend(clip, torch.zeros_like(clip), brightness_factor)


def adjust_saturation(clip: Tensor, saturation_factor: float) -> Tensor:
    """
    Args:
        clip (Tensor): Clip to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. Can be any
            non negative number. 0 gives a black and white image, 1 gives the
            original image while 2 enhances the saturation by a factor of 2.

    Returns:
        Tensor: Saturation adjusted image.
    """
    if saturation_factor < 0:
        raise ValueError('saturation_factor ({}) is not non-negative.'.format(saturation_factor))

    if not _is_tensor_a_torch_image(clip):
        raise TypeError('tensor is not a torch image.')

    return _blend(clip, rgb_to_grayscale_video(clip), saturation_factor)


def adjust_gamma(clip: Tensor, gamma: float, gain: float = 1) -> Tensor:
    r"""
    Also known as Power Law Transform. Intensities in RGB mode are adjusted
    based on the following equation:

    .. math::
        `I_{\text{out}} = 255 \times \text{gain} \times \left(\frac{I_{\text{in}}}{255}\right)^{\gamma}`

    See `Gamma Correction`_ for more details.

    .. _Gamma Correction: https://en.wikipedia.org/wiki/Gamma_correction

    Args:
        clip (Tensor): Tensor of RBG values to be adjusted.
        gamma (float): Non negative real number, same as :math:`\gamma` in the equation.
            gamma larger than 1 make the shadows darker,
            while gamma smaller than 1 make dark regions lighter.
        gain (float): The constant multiplier.
    """

    if not isinstance(clip, torch.Tensor):
        raise TypeError('Input img should be a Tensor.')

    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')

    result = clip
    dtype = clip.dtype
    if not torch.is_floating_point(clip):
        result = convert_image_dtype(result, torch.float32)

    result = (gain * result ** gamma).clamp(0, 1)

    result = convert_image_dtype(result, dtype)
    result = result.to(dtype)
    return result


def adjust_contrast(clip: Tensor, contrast_factor: float) -> Tensor:
    """Args:
        clip (Tensor): Clip to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.

    Returns:
        Tensor: Contrast adjusted image.
    """
    if contrast_factor < 0:
        raise ValueError('contrast_factor ({}) is not non-negative.'.format(contrast_factor))

    if not _is_tensor_a_torch_image(clip):
        raise TypeError('tensor is not a torch image.')

    dtype = clip.dtype if torch.is_floating_point(clip) else torch.float32
    mean = torch.mean(rgb_to_grayscale_video(clip).to(dtype), dim=(-4, -2, -1), keepdim=True)

    return _blend(clip, mean, contrast_factor)

def adjust_hue_video(clip: Tensor, hue_factor: float) -> Tensor:
    """
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.

    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.

    See `Hue`_ for more details.

    .. _Hue: https://en.wikipedia.org/wiki/Hue

    Args:
        clip (Tensor): Clip to be adjusted. Image type is either uint8 or float.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.

    Returns:
         Tensor: Hue adjusted image.
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))

    if not (isinstance(clip, torch.Tensor) and _is_tensor_a_torch_image(clip)):
        raise TypeError('Input img should be Tensor image')

    orig_dtype = clip.dtype
    if clip.dtype == torch.uint8:
        clip = clip.to(dtype=torch.float32) / 255.0

    clip = _rgb2hsv_video(clip)
    h, s, v = clip.unbind(dim=-4)
    h = (h + hue_factor) % 1.0
    clip = torch.stack((h, s, v), dim=-4)
    img_hue_adj = _hsv2rgb_video(clip)

    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)

    return img_hue_adj




