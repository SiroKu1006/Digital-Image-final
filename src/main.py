# file: src/main.py

import os
import argparse
import numpy as np
from utils import load_image, save_image, generate_gaussian_psf
from deblur_wiener import deblur_wiener
from deblur_rl import deblur_rl
from postprocess import denoise, enhance_edges
from metrics import compute_psnr, compute_ssim

def parse_args():
    parser = argparse.ArgumentParser(description="影像去模糊處理主程式")
    parser.add_argument(
        "-i", "--input", type=str, required=True,
        help="輸入模糊影像路徑 (e.g., data/blurred_images/blur1.png)"
    )
    parser.add_argument(
        "-o", "--output", type=str, default="data/results",
        help="輸出結果資料夾 (預設: data/results)"
    )
    parser.add_argument(
        "-m", "--method", type=str, choices=["wiener", "rl"], default="rl",
        help="去模糊方法: 'wiener' (Wiener Filter) 或 'rl' (Richardson-Lucy)，預設 'rl'"
    )
    parser.add_argument(
        "--psf_size", type=int, default=5,
        help="PSF (核) 大小 (方形核邊長)，預設 5"
    )
    parser.add_argument(
        "--psf_sigma", type=float, default=1.0,
        help="生成高斯 PSF 時的標準差 (sigma)，預設 1.0"
    )
    # Wiener 參數
    parser.add_argument(
        "--K", type=float, default=0.01,
        help="Wiener Filter 雜訊功率比 K，僅對 method='wiener' 生效，預設 0.01"
    )
    # Richardson-Lucy 參數
    parser.add_argument(
        "--iterations", type=int, default=30,
        help="Richardson-Lucy 迭代次數，僅對 method='rl' 生效，預設 30"
    )
    # 後處理參數 (可選)
    parser.add_argument(
        "--denoise_method", type=str, choices=["none", "median", "nl_means"], default="none",
        help="後處理去噪方法: 'none', 'median', 'nl_means'，預設 'none'"
    )
    parser.add_argument(
        "--denoise_radius", type=int, default=1,
        help="若 denoise_method='median'，中值濾波半徑，預設 1"
    )
    parser.add_argument(
        "--nl_patch_size", type=int, default=5,
        help="若 denoise_method='nl_means'，patch_size，預設 5"
    )
    parser.add_argument(
        "--nl_patch_distance", type=int, default=6,
        help="若 denoise_method='nl_means'，patch_distance，預設 6"
    )
    parser.add_argument(
        "--nl_h", type=float, default=None,
        help="若 denoise_method='nl_means'，過濾強度 h (若 None，則自動估算)，預設 None"
    )
    parser.add_argument(
        "--enhance_edges", action="store_true",
        help="是否對去模糊後影像進行邊緣增強 (unsharp mask)"
    )
    parser.add_argument(
        "--enhance_amount", type=float, default=1.0,
        help="邊緣增強強度 amount，預設 1.0"
    )
    parser.add_argument(
        "--enhance_radius", type=float, default=1.0,
        help="邊緣增強半徑 radius，預設 1.0"
    )
    # 評估指標參數 (可選)
    parser.add_argument(
        "--gt", type=str, default=None,
        help="若有銳利 (Ground Truth) 影像，可指定其路徑，將計算 PSNR/SSIM"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. 讀取影像
    print(f"讀取模糊影像: {args.input}")
    img = load_image(args.input, as_gray=True)

    # 2. 產生高斯 PSF
    print(f"生成高斯 PSF: 大小={args.psf_size}x{args.psf_size}, sigma={args.psf_sigma}")
    psf = generate_gaussian_psf(ksize=args.psf_size, sigma=args.psf_sigma)

    # 3. 去模糊
    if args.method == "wiener":
        print(f"使用 Wiener Filter 去模糊 (K={args.K})")
        deblurred = deblur_wiener(img, psf, K=args.K)
    else:
        print(f"使用 Richardson-Lucy 去模糊 (iterations={args.iterations})")
        deblurred = deblur_rl(img, psf, iterations=args.iterations)

    # 4. 後處理 (可選)
    if args.denoise_method != "none":
        print(f"後處理: 去噪 ({args.denoise_method})")
        if args.denoise_method == "median":
            deblurred = denoise(deblurred, method="median", radius=args.denoise_radius)
        else:
            h_val = args.nl_h if args.nl_h is not None else None
            deblurred = denoise(
                deblurred, method="nl_means",
                patch_size=args.nl_patch_size, patch_distance=args.nl_patch_distance, h=h_val
            )
    if args.enhance_edges:
        print(f"後處理: 邊緣增強 (amount={args.enhance_amount}, radius={args.enhance_radius})")
        deblurred = enhance_edges(deblurred, amount=args.enhance_amount, radius=args.enhance_radius)

    # 5. 儲存結果
    #    組合輸出目錄與檔名
    os.makedirs(args.output, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    out_name = f"{base_name}_{args.method}.png"
    out_path = os.path.join(args.output, out_name)
    print(f"儲存去模糊後影像: {out_path}")
    save_image(deblurred, out_path)

    # 6. 評估指標 (可選)
    if args.gt:
        print(f"計算評估指標，參考 Ground Truth: {args.gt}")
        gt_img = load_image(args.gt, as_gray=True)
        psnr_val = compute_psnr(gt_img, deblurred, data_range=1.0)
        ssim_val = compute_ssim(gt_img, deblurred, data_range=1.0)
        print(f"PSNR: {psnr_val:.2f} dB")
        print(f"SSIM: {ssim_val:.4f}")

if __name__ == "__main__":
    main()
