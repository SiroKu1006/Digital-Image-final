# file: src/postprocess.py

import numpy as np
from skimage import img_as_float
from skimage.filters import median, unsharp_mask
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, estimate_sigma

def denoise(img, method='median', **kwargs):
    """
    後處理：去除雜訊
    
    參數:
        img (numpy.ndarray): float 影像陣列 (範圍 0~1)
        method (str): 去噪方法 ('median' 或 'nl_means')
        **kwargs: 傳給對應去噪函式的其他參數
            - 如果 method='median'，可以傳入 radius (int)，預設為 1
            - 如果 method='nl_means'，可以傳入 patch_size, patch_distance, h (float) 等
    
    回傳:
        numpy.ndarray: 去噪後的影像 (float, 範圍 0~1)
    """
    img_float = img_as_float(img)
    
    if method == 'median':
        radius = kwargs.get('radius', 1)
        # 使用中值濾波 (median filter)
        denoised = median(img_float, disk(radius))
        return np.clip(denoised, 0, 1)
    
    elif method == 'nl_means':
        # 使用非局部平均 (NL-Means) 去噪
        # 先估算雜訊標準差
        sigma_est = estimate_sigma(img_float, multichannel=False)
        patch_size = kwargs.get('patch_size', 5)
        patch_distance = kwargs.get('patch_distance', 6)
        h = kwargs.get('h', 1.15 * sigma_est)
        denoised = denoise_nl_means(
            img_float,
            h=h,
            patch_size=patch_size,
            patch_distance=patch_distance,
            fast_mode=True
        )
        return np.clip(denoised, 0, 1)
    
    else:
        raise ValueError("Unknown denoising method: choose 'median' or 'nl_means'")

def enhance_edges(img, amount=1.0, radius=1.0):
    """
    後處理：增強邊緣 (使用 unsharp mask)
    
    參數:
        img (numpy.ndarray): float 影像陣列 (範圍 0~1)
        amount (float): 強度係數，預設 1.0
        radius (float): 半徑大小，預設 1.0
    
    回傳:
        numpy.ndarray: 增強邊緣後的影像 (float, 範圍 0~1)
    """
    img_float = img_as_float(img)
    # 使用 unsharp mask 增強邊緣
    sharpened = unsharp_mask(img_float, radius=radius, amount=amount)
    return np.clip(sharpened, 0, 1)
