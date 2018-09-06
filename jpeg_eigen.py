# -*- coding: UTF-8 -*-
"""
JPEG Implementation Forensics Based on Eigen-Algorithms
@author: Paolo Bestagini (paolo.bestagini@polimi.it)
"""
import os

import numpy as np
from PIL import Image, ExifTags
from scipy.fftpack import dct, idct
from skimage.util import view_as_blocks


class RecompressError(Exception):
    pass


def imread_orientation(img_in):
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    exif = dict(img_in._getexif().items())

    if exif is not None:
        if orientation in exif.keys():
            if exif[orientation] == 3:
                img_in = img_in.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img_in = img_in.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img_in = img_in.rotate(90, expand=True)

    return img_in


def jpeg_recompress_pil(img_path_in: str, img_path_out: str, img_shape: tuple = None, qtables_in=None,
                        check=False) -> None:
    """Re-compress a JPEG image using the same quantization matrix and PIL implementation.

    Args:
        img_path_in (str): path to input JPEG image.
        img_path_out (str): path to output JPEG image.
        qtables_in (np.array): quantization table to apply.
        check (bool): check input and output quantization tables.
    """
    # Read Data
    img_in = Image.open(img_path_in)
    if not qtables_in:
        qtables_in = img_in.quantization

    # Resize image
    if img_shape is not None:
        img_in = imread_orientation(img_in)
        if (img_in.size[0] >= img_in.size[1] and img_shape[0] < img_shape[1]) or (
                img_in.size[0] < img_in.size[1] and img_shape[0] >= img_shape[1]):
            img_shape = [img_shape[1], img_shape[0]]
            pass
        img_in = img_in.resize(img_shape, Image.LANCZOS)

    # Re-compress image
    os.makedirs(os.path.split(img_path_out)[0], exist_ok=True)
    img_in.save(img_path_out, format='JPEG', subsample='keep', qtables=qtables_in)

    # Check qtables
    if check:
        img_out = Image.open(img_path_out)
        qtables_out = img_out.quantization
        img_out.close()
        if qtables_in != qtables_out:
            raise RecompressError('Input and output quantization tables are different.')

    # Close
    img_in.close()


def compute_jpeg_dct_Y(img_Y: np.ndarray) -> np.ndarray:
    """Compute block-wise DCT in a JPEG-like fashion

    Args:
        img_Y (np.array): luminance component of input JPEG image.

    Returns:
        img_blocks_dct (np.array): block-wise DCT
    """
    # Parameters
    B = 8

    # Check B division and pad
    dH, dW = np.asarray(img_Y.shape) % B
    if dH != 0:
        dH = B - dH
    if dW != 0:
        dW = B - dW
    img_Y = np.pad(img_Y, ((0, dH), (0, dW)), mode='reflect')

    # Split Into Blocks
    img_blocks = view_as_blocks(img_Y, block_shape=(B, B))
    img_blocks = np.reshape(img_blocks, (-1, B, B))

    # Compute DCT
    img_blocks_dct = dct(dct(img_blocks, axis=1, norm='ortho'), axis=2, norm='ortho')

    return img_blocks_dct


def jpeg_compress_Y(img_Y: np.ndarray, qtable: np.ndarray, quant_fun: callable = np.round):
    """Simulate luminance component JPEG compression.

    Args:
        img_Y (np.array): luminance component of input JPEG image.
        qtable (np.array): JPEG quantization table.
        quant_fun (function): quantization function

    Returns:
        img_Y_comp (np.array): luminance component of output JPEG image.
    """
    # Parameters
    B = 8

    # Check B division and pad
    H, W = img_Y.shape
    dH, dW = np.asarray(img_Y.shape) % B
    if dH != 0:
        dH = B - dH
    if dW != 0:
        dW = B - dW
    img_Y = np.pad(img_Y, ((0, dH), (0, dW)), mode='reflect')

    # Compute DCT
    img_blocks_dct = compute_jpeg_dct_Y(img_Y - 128.)

    # Quantize and de-quantize
    img_blocks_dct_q = qtable * quant_fun(img_blocks_dct / qtable)

    # Compute IDCT
    img_blocks_idct = idct(idct(img_blocks_dct_q, axis=2, norm='ortho'), axis=1, norm='ortho')

    # Reshape
    img_Y_comp = np.zeros(img_Y.shape)
    i = 0
    for h in np.arange(0, img_Y.shape[0], B):
        for w in np.arange(0, img_Y.shape[1], B):
            img_Y_comp[h:h + B, w:w + B] = img_blocks_idct[i]
            i += 1
    img_Y_comp = np.clip(np.round(128. + img_Y_comp), 0, 255)
    img_Y_comp = img_Y_comp[:H, :W]

    return img_Y_comp


def jpeg_feature(img_path: str) -> np.ndarray:
    """Extract JPEG feature.

    Args:
        img_path (str): path to input JPEG image.

    Returns:
        feature (np.array): feature vector
    """
    # Params
    zig_zag_idx = [0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42,
                   3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40, 44, 53,
                   10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60,
                   21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63]

    # Init
    img = Image.open(img_path)
    img.draft('YCbCr', None)
    qtable = np.asarray(img.quantization[0])[zig_zag_idx].reshape((8, 8))

    # Original Image Data
    img_Y_0 = np.asarray(img, dtype=np.float32)[:, :, 0]
    img_blocks_dct_0 = compute_jpeg_dct_Y(img_Y_0 - 128.)

    # Loop over quantization functions
    quant_fun_list = [np.round,
                      lambda x: np.floor(x + 0.5),
                      lambda x: np.ceil(x - 0.5),
                      ]
    feature = np.zeros((len(quant_fun_list), 64))

    for q_idx, quant_fun in enumerate(quant_fun_list):
        # First JPEG
        img_Y_1 = jpeg_compress_Y(img_Y_0, qtable, quant_fun)
        img_blocks_dct_1 = compute_jpeg_dct_Y(img_Y_1 - 128.)

        # Second JPEG
        img_Y_2 = jpeg_compress_Y(img_Y_1, qtable, quant_fun)
        img_blocks_dct_2 = compute_jpeg_dct_Y(img_Y_2 - 128.)

        # Feature
        mse_single = np.mean((img_blocks_dct_0 - img_blocks_dct_1) ** 2, axis=0).reshape(-1)
        mse_double = np.mean((img_blocks_dct_1 - img_blocks_dct_2) ** 2, axis=0).reshape(-1)

        feature[q_idx] = (mse_double - mse_single) ** 2

    feature = np.concatenate(feature, axis=0)

    return feature
