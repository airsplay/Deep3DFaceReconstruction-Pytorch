import sys
sys.path.append(".")
import os

import numpy as np
import torch
from torch.optim import Adam
import tqdm

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

    # Coefficient initialization
    shape_coef = torch.zeros(80, device=device)
    exp_coef = torch.zeros(64, device=device)
    texture_coef = torch.zeros(80, device=device)
    angle_coef = torch.zeros(3, device=device)
    gamma_coef = torch.zeros(27, device=device)
    translation_coef = torch.zeros(3, device=device)

    # Optimization Setup
    optimized_coefs = [
        shape_coef,
    ]
    for coef in optimized_coefs:
        coef.requires_grad = True
    optim = Adam(optimized_coefs, lr=1e-1)

    num_steps = 1000
    for step in tqdm.tqdm(range(num_steps)):
        arr_coef = [
            shape_coef,
            exp_coef,
            texture_coef,
            angle_coef,
            gamma_coef,
            translation_coef,
        ]
        coef = torch.cat(arr_coef, 0).unsqueeze(0).repeat(11, 1)

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

        loss = images.pow(2).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

    images = images.detach().cpu().numpy()
    images = np.squeeze(images)

    path_str = "output/optim_test.png"
    path = os.path.split(path_str)[0]
    if os.path.exists(path) is False:
        os.makedirs(path)

    from PIL import Image
    images = np.uint8(images[:, :, :3] * 255.0)
    img = Image.fromarray(images)
    img.save(path_str)

    face_shape = face_shape.detach().cpu().numpy()
    face_color = face_color.detach().cpu().numpy()

    face_shape = np.squeeze(face_shape)
    face_color = np.squeeze(face_color)
    save_obj(path_str + '.obj', face_shape, bfm.tri, np.clip(face_color, 0, 1.0))  # 3D reconstruction face (in canonical view)

if __name__ == '__main__':
    recon()
