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
Alternative installation option (the above needs some additional library):
```
pip install -r requirements.txt
pip install torch==1.7.0+cu101 torchvision==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py36_cu101_pyt170/download.html
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
python test/test_recon.py
```



## Features

### MTCNN

I use mtcnn to crop raw images and detect 5 landmarks. The most code of MTCNN comes from [FaceNet-pytorch](https://github.com/timesler/facenet-pytorch).

### Pytorc3d

In this repo, I use [PyTorch3d 0.3.0](https://github.com/facebookresearch/pytorch3d) to render the reconstructed images.

### Estimating Intrinsic Parameters

In the origin repo ([Deep3DFaceReconstruction-pytorch](https://github.com/changhongjian/Deep3DFaceReconstruction-pytorch)), the rendered images is not the same as the input image because of `preprocess`. So, I add the `estimate_intrinsic` to get intrinsic parameters.

## Examples:

Here are some examples:

|Origin Images|Cropped Images|Rendered Images|
|-------------|---|---|
|![Putin](examples/origin.jpg)|![Putin](examples/cropped.jpg)|![putin](examples/rendered.png)|


## File Architecture

```
├─BFM               same as Deep3DFaceReconstruction
├─dataset           storing the corpped images
│  └─Vladimir_Putin
├─examples          show examples
├─facebank          storing the raw/origin images
│  └─Vladimir_Putin
├─models            storing the pretrained models
├─output            storing the output images(.mat, .png)
│  └─Vladimir_Putin
└─preprocess        cropping images and detecting landmarks
    ├─data          storing the models of mtcnn
    ├─utils
```

Also, this repo can also generate the UV map, and you need download UV coordinates from the following link:  
&nbsp;&nbsp;Download UV coordinates fom STN website: https://github.com/anilbas/3DMMasSTN/blob/master/util/BFM_UV.mat  
&nbsp;&nbsp;Copy BFM_UV.mat to BFM

The pretrained models can be downloaded from [Google Drive](https://drive.google.com/file/d/1JjLl8-7Qurwlq5q61hSJEbCKFrhPh0t2/view?usp=sharing).
