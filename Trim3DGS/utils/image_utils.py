#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def generate_grid(x_low, x_high, x_num, y_low, y_high, y_num, device):
	xs = torch.linspace(x_low, x_high, x_num, device=device)
	ys = torch.linspace(y_low, y_high, y_num, device=device)
	xv, yv = torch.meshgrid([xs, ys], indexing='xy')
	grid = torch.stack((xv.flatten(), yv.flatten())).T
	return grid

def interpolate(data, xy):
    """
    Interpolates values from a grid of data based on given coordinates.

    Args:
        data (torch.Tensor): The input data grid of shape (..., H, W).
        xy (torch.Tensor): The coordinates to interpolate the values from, of shape (N, 2). The coordinates are
            expected to be in the range [0, 1].

    Returns:
        torch.Tensor: The interpolated values of shape (..., N).
    """

    pos = xy * torch.tensor([data.shape[-1], data.shape[-2]], dtype=torch.float32, device=xy.device)
    indices = pos.long()
    lerp_weights = pos - indices.float()
    x0 = indices[:, 0].clamp(min=0, max=data.shape[-1]-1)
    y0 = indices[:, 1].clamp(min=0, max=data.shape[-2]-1)
    x1 = (x0 + 1).clamp(max=data.shape[-1]-1)
    y1 = (y0 + 1).clamp(max=data.shape[-2]-1)

    return (
        data[..., y0, x0] * (1.0 - lerp_weights[:,0]) * (1.0 - lerp_weights[:,1]) +
        data[..., y0, x1] * lerp_weights[:,0] * (1.0 - lerp_weights[:,1]) +
        data[..., y1, x0] * (1.0 - lerp_weights[:,0]) * lerp_weights[:,1] +
        data[..., y1, x1] * lerp_weights[:,0] * lerp_weights[:,1]
    )

def compute_gradient(image, RGB2GRAY=False):
    assert image.ndim == 4, "image must have 4 dimensions"
    assert image.shape[1] == 1 or image.shape[1] == 3, "image must have 1 or 3 channels"
    if image.shape[1] == 3:
        assert RGB2GRAY == True, "RGB image must be converted to grayscale first"
        image = rgb_to_gray(image)
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)
    sobel_kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

    image_for_pad = F.pad(image, pad=(1, 1, 1, 1), mode="replicate")
    gradient_x = F.conv2d(image_for_pad, sobel_kernel_x) / 3
    gradient_y = F.conv2d(image_for_pad, sobel_kernel_y) / 3

    return gradient_x, gradient_y

def rgb_to_gray(image):
    gray_image = (0.299 * image[:, 0, :, :] + 0.587 * image[:, 1, :, :] +
                  0.114 * image[:, 2, :, :])
    gray_image = gray_image.unsqueeze(1)

    return gray_image

def blur(image):
    if image.ndim == 2:
        image = image[None, None, ...]
    channel = image.shape[1]
    guassian_kernel = torch.tensor([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]], dtype=image.dtype, device=image.device).view(1, 1, 3, 3).repeat(channel, 1, 1, 1)
    output = F.conv2d(image, guassian_kernel, padding=1, groups=channel)
    return output

import matplotlib.pyplot as plt
import numpy as np

def draw_hist(tensor, path, density=False):
    tensor = tensor.reshape(-1).detach().cpu().numpy()
    _ = plt.hist(tensor, bins=50, density=density)
    plt.savefig(path)