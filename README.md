# Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set

## Preparation
1. Install python env (after building the virtualenv).
   ```
   pip install -r requirements.txt
   pip install torch==1.7.0+cu101 torchvision==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
   pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py36_cu101_pyt170/download.html
   ```
If the CUDA version does not match, we could change the `torch==1.7.0+cu101` to
  - `torch==1.7.0` for CUDA 10.0
  - `torch==1.7.0+cu110` for CUDA 11.0.
  - `torch==1.7.0+cu92` for CUDA 9.2 
    If all above  do not work, directly compile pytorch3d with latest PyTorch (need [Nvidia/cub](https://github.com/NVIDIA/cub))
   ```
   pip install -r requirements.txt
   pip install torch
   pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
   ```
2. Download the front-face BFM model (already preprocessed)
   ```shell
   wget https://nlp.cs.unc.edu/data/BFM_model_front.mat -P BFM/ 
   ```
3. install CLIP
   ```shell
   pip install ftfy regex tqdm
   # Do not upgrade PyTorch version so PyTorch3D would work:
   pip install --no-deps git+https://github.com/openai/CLIP.git  
   ```


## Running
```shell
python bfmface/main.py -p "Abraham Lincoln with a big mouse | real human"
```
