# Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set

Pytorch version of the repo [Deep3DFaceReconstruction](https://github.com/microsoft/Deep3DFaceReconstruction).

This repo only contains the **reconstruction** part, so you can use [Deep3DFaceReconstruction-pytorch](https://github.com/changhongjian/Deep3DFaceReconstruction-pytorch) repo to train the network. And the pretrained model is also from this [repo](https://github.com/changhongjian/Deep3DFaceReconstruction-pytorch/tree/master/network).


## Preparation
1. install env (after building the virtualenv). Directly compile pytorch3d with latest PyTorch (need [Nvidia/cub](https://github.com/NVIDIA/cub))
```
pip install -r requirements.txt
pip install torch
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
Alternative Linux installation option (the above needs some additional library):
```
pip install -r requirements.txt
pip install torch==1.7.0+cu101 torchvision==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py36_cu101_pyt170/download.html
```
Mac installation:
```
pip install -r requirements.txt
pip install torch==1.7.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision
pip install pytorch3d 
```


2. Download BFM at [link](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads), put "01_MorphableModel.mat" into ./BFM subfolder. 

3. Download Expression Basis by Guo et al. at [link](https://github.com/Juyong/3DFace). Download the "CoarseData" link. Put "Exp_Pca.bin" into ./BFM subfolder.

4. Download model weight from [Google Drive](https://drive.google.com/file/d/1JjLl8-7Qurwlq5q61hSJEbCKFrhPh0t2/view?usp=sharing). Put params/network/params.pt under ./models/.

5. Download https://github.com/anilbas/3DMMasSTN/blob/master/util/BFM_UV.mat to ./BFM.
```
cd BFM
wget https://github.com/anilbas/3DMMasSTN/blob/master/util/BFM_UV.mat?raw=true -O BFM_UV.mat
cd ..
```


## Test
1. Test reconstruction code:
```
python test/recon_test.py
```
2. Test modeling and rendering:
```shell
python test/modeling_test.py
```


