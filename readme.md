



##初始化
```
python -m venv venv
venv\Scripts\activate
pip install numpy scipy scikit-image matplotlib opencv-python
```

## 執行
### rl (Richardson–Lucy 演算法)
```
cd project/src
python main.py \
  -i data/case1.jpg \
  -o results \
  -m rl \
  --psf_size 5 \
  --psf_sigma 1.0 \
  --iterations 30
```
- -i data/case1.jpg
    輸入模糊影像路徑，這裡指向 src/data/ 下的 case1.jpg。

- -o results
    去模糊後影像輸出資料夾，程式會在 src/results/ 下建立並存放結果。

- -m rl
    去模糊方法：rl 表示使用 Richardson–Lucy 演算法；若改成 wiener 則使用 Wiener Filter。

- --psf_size 5
    PSF（Point Spread Function，模糊核）為 5×5；必須為正整數且為奇數。

- --psf_sigma 1.0
    生成高斯 PSF 時所使用的標準差 (σ)，可根據影像模糊程度調整。

- --iterations 30
    僅對 rl 方法生效，指定 Richardson–Lucy 迭代次數為 30。迭代次數越高，還原效果較強，但同時運算時間也會增加。

### wiener (Wiener Filter)
```
cd src
python main.py \
  -i data/case2.png \
  -o results \
  -m wiener \
  --psf_size 7 \
  --psf_sigma 1.5 \
  --K 0.02
```
- -m wiener
使用 Wiener Filter 去模糊。

- --K 0.02
Wiener Filter 的雜訊功率比 (noise-to-signal ratio)，通常介於 0.001～0.1，K 值越小，去模糊越強，但雜訊鬆弛 (ringing) 現象也會越明顯；K 值越大，則去模糊效果較弱。
