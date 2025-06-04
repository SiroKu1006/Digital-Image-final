import torch
import torch.nn.functional as F
import numpy as np

def to_tensor(img_np):
    """將 numpy (H, W, C) 轉為 tensor (1, C, H, W)，並放到裝置上"""
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0)
    return img_tensor


def to_numpy(tensor):
    """將 tensor (1, C, H, W) 轉為 numpy (H, W, C)，並裁剪至 0~1"""
    return tensor.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)


def fft_convolve2d(img, kernel):
    """使用 FFT 進行 2D 卷積"""
    padding = [k // 2 for k in kernel.shape[-2:]]
    img_fft = torch.fft.rfft2(img, dim=(-2, -1))
    kernel_fft = torch.fft.rfft2(kernel, s=img.shape[-2:], dim=(-2, -1))
    return torch.fft.irfft2(img_fft * kernel_fft, s=img.shape[-2:])


def deblur_rl(img_np, psf_np, iterations=30):
    """
    使用 GPU 加速的 Richardson-Lucy 演算法

    img_np: numpy.ndarray, shape=(H, W, 3), RGB, range 0~1
    psf_np: numpy.ndarray, shape=(k, k), normalized PSF
    return: numpy.ndarray, shape=(H, W, 3), RGB, range 0~1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = to_tensor(img_np).to(device)        # [1, 3, H, W]
    estimate = img.clone()                    # 初始估計值

    # 準備 PSF
    psf = torch.from_numpy(psf_np).float().to(device)
    psf = psf.unsqueeze(0).unsqueeze(0)       # [1, 1, k, k]
    psf = psf.repeat(3, 1, 1, 1)              # [3, 1, k, k] -> 適用每個 channel

    for _ in range(iterations):
        estimate.requires_grad = False

        blurred = F.conv2d(estimate, psf, padding='same', groups=3)

        blurred = torch.clamp(blurred, min=1e-6)

        # 更新
        ratio = img / blurred
        correction = F.conv2d(ratio, torch.flip(psf, dims=[2, 3]), padding='same', groups=3)
        estimate *= correction

    return to_numpy(estimate)
