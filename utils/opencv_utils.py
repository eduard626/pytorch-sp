#!/usr/bin/env python

import cv2
import numpy as np
from pathlib import Path

def gaussian_blur(img : np.array, kernel_size : tuple =(5,5), sigma_x : float = 0.5, sigma_y : float =0.) -> np.array:
    """ Aux wrapper for opencv's gaussian blur

    Args:
        img (np.array): The image to blur
        kernel_size (tuple, optional): Size/shape of the kernel. Defaults to (5,5).
        sigma_x (float, optional): Std deviation. Defaults to 0.5.

    Returns:
        np.array: Blurred image
    """
    assert img.ndim == 2, "Image must be single channel. Recieved {}".format(img.shape)
    out = cv2.GaussianBlur(img, kernel_size, sigmaX=sigma_x, sigmaY=sigma_y)


def resize_image(img : np.array, target_size : tuple(0,0), sx : float = 0.5, sy : float = 0.5, interp_method :int = cv2.INTER_LINEAR) -> np.array:
    """ Wrapper for cv.resize to resize an image

    Args:
        img (np.array): The image to resize
        target_size (tuple): Target size for output
        sx (float, optional): Scale factor x_. Defaults to 0.5.
        sy (float, optional): Scale factor y . Defaults to 0.5.
        interp_method (int, optional): Interpolation method. Defaults to cv2.INTER_LINEAR.

    Returns:
        np.array: Resized image
    """
    return cv2.resize(img, dsize=target_size, fx=sx, fy=sy, interpolation=interp_method)

def write_image(img : np.array, img_path : Path) -> bool:
    assert img_path.parent.exists(), "Sent an invalid path {}".format(img_path)
    return cv2.imwrite(str(img_path), img)