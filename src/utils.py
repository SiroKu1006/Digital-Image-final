# file: src/utils.py

import numpy as np
from skimage import io, img_as_float
import cv2

def load_image(path, as_gray=True):
    """
    讀取影像並轉換成 float 型態（範圍 0~1）
    
    參數:
        path (str): 影像檔案路徑
        as_gray (bool): 如果為 True，則將影像轉為灰階

    回傳:
        numpy.ndarray: 以 float 型態儲存、範圍為 [0, 1] 的影像矩陣
    """
    img = io.imread(path, as_gray=as_gray)
    return img_as_float(img)

def save_image(img, path):
    """
    將 float 影像 (範圍 0~1) 存成 uint8 檔案
    
    參數:
        img (numpy.ndarray): 要存檔的影像 (float, 範圍 0~1)
        path (str): 輸出影像檔案路徑
    """
    # 將影像轉換成 uint8
    img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    io.imsave(path, img_uint8)

def generate_gaussian_psf(ksize=5, sigma=1.0):
    """
    產生一個 ksize x ksize 的高斯 PSF (normalized)
    
    參數:
        ksize (int): PSF 核的邊長 (正方形)
        sigma (float): 高斯標準差
    
    回傳:
        numpy.ndarray: 已 normalize 的高斯核 (sum 約為 1)
    """
    # 使用 OpenCV 生成一維高斯核，然後外積變成二維
    gk = cv2.getGaussianKernel(ksize, sigma)
    psf = gk @ gk.T
    psf = psf / psf.sum()
    return psf
