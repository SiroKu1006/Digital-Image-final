import numpy as np
from skimage.restoration import richardson_lucy

def deblur_rl(img, psf, iterations=30):
    """
    使用 Richardson–Lucy 演算法去模糊影像

    參數:
        img (numpy.ndarray): float 影像陣列 (範圍 0~1)
        psf (numpy.ndarray): float PSF 核 (已 normalize)
        iterations (int): 迭代次數，預設 30

    回傳:
        numpy.ndarray: 去模糊後的影像 (float, 範圍 0~1)
    """
    if img.ndim == 3:
        channels = []
        for c in range(3):
            channel = richardson_lucy(img[..., c], psf, num_iter=iterations)
            channels.append(np.clip(channel, 0, 1))
        return np.stack(channels, axis=-1)
    else:
        return np.clip(richardson_lucy(img, psf, num_iter=iterations), 0, 1)