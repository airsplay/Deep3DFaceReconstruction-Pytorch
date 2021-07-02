import sys
sys.path.append(".")

import os
import glob
import torch
import numpy as np
from models.resnet_50 import resnet50_use
from load_data import transfer_BFM09, BFM, load_img, Preprocess, save_obj
from modeling_3d import reconstruction
from rendering import render_img



def recon():
    # read BFM face model
    # transfer original BFM model to our model
    if not os.path.isfile('BFM/BFM_model_front.mat'):
        transfer_BFM09()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    bfm = BFM(r'BFM/BFM_model_front.mat', device)

    
    arr_coef = [
        torch.zeros(1, 80+64+80, device=device),
        torch.zeros(1, 3, device=device),
        torch.zeros(1, 27, device=device),
        torch.zeros(1, 3, device=device),
    ]
    coef = torch.cat(arr_coef, 1)
    coef.requires_grad = True


    # reconstruct 3D face with output coefficients and face model
    face_shape, face_texture, face_color, angles, translation, gamma = reconstruction(coef, bfm)

    # Use the default intrinsic from Deep3DFaceRecon: https://github.com/microsoft/Deep3DFaceReconstruction/blob/master/demo.py
    fx = 1015.0
    fy = 1015.0
    px = 0.
    py = 0.
    intrinsic = (fx, fy, px, py)

    # We test with an extrinsic here
    R = torch.eye(3, device=device).unsqueeze(0)
    T = torch.zeros(3, device=device).unsqueeze(0)
    extrinsic = (R, T)

    # Since the camera model use Rx + T, we want to do some transpotation first, resulting in R (x + T0) + T.
    face_shape[:, :, 2] = 10.0 - face_shape[:, :, 2]

    # Normalize color space from [0, 255] --> [0, 1]
    face_color = face_color / 255.0

    # Rendering the image
    images = render_img(face_shape, face_color, bfm, 300, extrinsic, intrinsic)
    images.sum().backward()

    print(coef.grad)

    images = images.detach().cpu().numpy()
    images = np.squeeze(images)

    path_str = "output/test.png"
    path = os.path.split(path_str)[0]
    if os.path.exists(path) is False:
        os.makedirs(path)

    from PIL import Image
    images = np.uint8(images[:, :, :3] * 255.0)
    # init_img = np.array(img)
    # init_img[images != 0] = 0
    # images += init_img
    img = Image.fromarray(images)
    img.save(path_str)

    face_shape = face_shape.detach().cpu().numpy()
    face_color = face_color.detach().cpu().numpy()

    face_shape = np.squeeze(face_shape)
    face_color = np.squeeze(face_color)
    #save_obj(, '_mesh.obj'), face_shape, bfm.tri, np.clip(face_color, 0, 1.0))  # 3D reconstruction face (in canonical view)

if __name__ == '__main__':
    recon()
