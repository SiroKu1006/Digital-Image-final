import numpy as np
from skimage.restoration import wiener

def deblur_wiener(img, psf, K=0.01):
    """
    使用 Wiener Filter 去模糊影像

    參數:
        img (numpy.ndarray): float 影像陣列 (範圍 0~1)
        psf (numpy.ndarray): float PSF 核 (已 normalize)
        K (float): 雜訊功率比 (noise-to-signal ratio)，預設 0.01

    回傳:
        numpy.ndarray: 去模糊後的影像 (float, 範圍 0~1)
    """
    # 呼叫 skimage.restoration.wiener 進行去模糊
    deblurred = wiener(img, psf, K)
    # 確保輸出值在 [0, 1] 範圍內
    deblurred = np.clip(deblurred, 0, 1)
    return deblurred