# file: src/metrics.py

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_psnr(orig, restored, data_range=1.0):
    """
    計算 PSNR (Peak Signal-to-Noise Ratio)

    參數:
        orig (numpy.ndarray): 原始銳利影像 (float 或 uint8)
        restored (numpy.ndarray): 去模糊後影像 (float 或 uint8)
        data_range (float 或 int): 影像的資料範圍 (若影像為 float 範圍在 [0,1]，則 data_range=1.0;
                                     若影像為 uint8 範圍在 [0,255]，則 data_range=255)

    回傳:
        float: PSNR 值 (越高代表相似度越高)
    """
    # 確保兩張影像維度相同
    if orig.shape != restored.shape:
        raise ValueError("原始影像與還原影像尺寸不同，無法計算 PSNR")

    # 若輸入為 float，自動使用 data_range; 若為 uint8，可指定 data_range=255
    return psnr(orig, restored, data_range=data_range)

def compute_ssim(orig, restored, data_range=1.0):
    """
    計算 SSIM (Structural Similarity Index)

    參數:
        orig (numpy.ndarray): 原始銳利影像 (float 或 uint8)
        restored (numpy.ndarray): 去模糊後影像 (float 或 uint8)
        data_range (float 或 int): 影像的資料範圍 (若影像為 float 範圍在 [0,1]，則 data_range=1.0;
                                     若影像為 uint8 範圍在 [0,255]，則 data_range=255)

    回傳:
        float: SSIM 值 (範圍 [-1, 1]，越接近 1 表示結構越相似)
    """
    # 確保兩張影像維度相同
    if orig.shape != restored.shape:
        raise ValueError("原始影像與還原影像尺寸不同，無法計算 SSIM")

    # multichannel: 若為彩色影像 (shape = (H, W, C))，可設 multichannel=True
    multichannel = True if orig.ndim == 3 else False

    return ssim(orig, restored, data_range=data_range, multichannel=multichannel)
