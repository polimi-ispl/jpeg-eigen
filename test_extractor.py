# -*- coding: UTF-8 -*-
"""
JPEG Implementation Forensics Based on Eigen-Algorithms
@author: Paolo Bestagini (paolo.bestagini@polimi.it)
@author: Luca Bondi (luca.bondi@polimi.it)
"""
import os
import unittest

import numpy as np
from PIL import Image

from jpeg_eigen import jpeg_recompress_pil, jpeg_feature


class TestExtractor(unittest.TestCase):
    im_png_path = 'samples/raw.png'
    im_ps_path = 'samples/photoshop.jpg'
    im_ps_pil_path = 'samples/photoshop_pil.jpg'
    im_pil_path = 'samples/pil.jpg'

    def test_extract(self):

        # double compression
        if os.path.exists(self.im_ps_pil_path):
            os.unlink(self.im_ps_pil_path)
        jpeg_recompress_pil(self.im_ps_path, self.im_ps_pil_path, check=True)
        self.assertTrue(os.path.exists(self.im_ps_pil_path))

        # single compression
        img_in = Image.open(self.im_ps_path)
        qtables_in = img_in.quantization
        if os.path.exists(self.im_pil_path):
            os.unlink(self.im_pil_path)
        jpeg_recompress_pil(self.im_png_path, self.im_pil_path, qtables_in=qtables_in, check=True)
        self.assertTrue(os.path.exists(self.im_pil_path))

        ps_features = jpeg_feature(self.im_ps_path)
        pil_features = jpeg_feature(self.im_pil_path)
        ps_pil_features = jpeg_feature(self.im_ps_pil_path)

        ps_features_ref = np.load('samples/ps_features.npy')
        pil_features_ref = np.load('samples/pil_features.npy')
        ps_pil_features_ref = np.load('samples/ps_pil_features.npy')

        self.assertTrue(np.allclose(ps_features_ref, ps_features))
        self.assertTrue(np.allclose(pil_features_ref, pil_features))
        self.assertTrue(np.allclose(ps_pil_features_ref, ps_pil_features))
