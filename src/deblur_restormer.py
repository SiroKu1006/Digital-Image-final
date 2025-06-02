# file: src/deblur_restormer.py

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# 載入 Restormer 模型（需先安裝）
from restormer_arch import Restormer
from basicsr.utils.download_util import load_file_from_url

def load_restormer_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 改為 defocus deblurring 模型路徑
    model_path = os.path.join(r'C:\Users\peipe\OneDrive\桌面\MyCode\Digital-Image-final\src\.cache\restormer\single_image_defocus_deblurring.pth')
    
    if not os.path.exists(model_path):
        print(f"找不到模型檔案：{model_path}")
        return None
    
    model = Restormer()
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['params'])
    model.eval()
    
    return model

def deblur_restormer(img_np):
    """
    img_np: numpy.ndarray, shape=(H, W, 3), RGB, range 0~1
    return: numpy.ndarray, shape=(H, W, 3), RGB, range 0~1
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_restormer_model().to(device)

    # 前處理: to tensor
    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        output = model(img_tensor)
        out_np = output[0].cpu().squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0)

    return out_np
