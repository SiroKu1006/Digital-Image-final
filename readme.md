
##初始化
```
python -m venv venv
git clone https://github.com/swz30/Restormer.git
venv\Scripts\activate

pip install pytorch=1.8 torchvision cudatoolkit=10.2 -c pytorch
pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm
pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips

python Restormer/setup.py develop --no_cuda_ext