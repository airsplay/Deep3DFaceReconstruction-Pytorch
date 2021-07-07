import sys
sys.path.append(".")
import os
import math

import numpy as np
import torch
from torch.optim import Adam
import tqdm

from load_data import transfer_BFM09, BFM, load_img, Preprocess, save_obj
from modeling_3d import reconstruction, compute_rotation_matrix
from loss import ClipLoss
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
    optim = Adam(optimized_coefs, lr=1e-2)

    criterion = ClipLoss(prompt="human face").to(device)

    num_steps = 1000
    image_size = 224
    for step in tqdm.tqdm(range(num_steps)):
        arr_coef = [
            shape_coef,
            exp_coef,
            texture_coef,
            angle_coef,
            gamma_coef,
            translation_coef,
        ]
        coef = torch.cat(arr_coef, 0).unsqueeze(0)      # coef of shape [1, #coef_code]

        # reconstruct 3D face with BFM coefficients and face model
        face_shape, face_texture, face_color, angles, translation, gamma = reconstruction(coef, bfm)
        face_idx = torch.tensor(bfm.tri, device=device).unsqueeze(0) - 1                    # index in BFM start from 1, we minus 1 here.
        face_color = face_color / 255.0             # Normalize color space from [0, 255] --> [0, 1]

        # Camera extrinsics
        psi_range = math.pi / 6
        psi_samples = 3
        theta_range = math.pi / 3
        theta_samples = 3
        psi = torch.linspace(-psi_range, psi_range, psi_samples, device=device)
        theta = torch.linspace(-theta_range, theta_range, theta_samples, device=device)
        num_cameras = psi_samples * theta_samples
        phi = torch.zeros(1, device=device)
        euler = torch.cartesian_prod(psi, theta, phi)
        camera_direction = compute_rotation_matrix(euler)
        camera_distance = 10.
        R_y_flip = torch.tensor(
            [[-1, 0, 0],
             [0, 1, 0],
             [0, 0, -1]],
            dtype=torch.float,
            device=device,
        )
        R_inverse_camera_rotation = compute_rotation_matrix(-euler)
        R = torch.matmul(R_y_flip, R_inverse_camera_rotation)
        T = torch.tensor(
            [0, 0, camera_distance],
            dtype=torch.float,
            device=device
        ).repeat(num_cameras, 1)
        extrinsic = (R, T)

        # Use the default intrinsic from Deep3DFaceRecon:
        #   https://github.com/microsoft/Deep3DFaceReconstruction/blob/master/demo.py
        focus_length_in_screen = 1015.0
        fx = torch.full((num_cameras,), focus_length_in_screen, device=device)
        fy = torch.full((num_cameras,), focus_length_in_screen, device=device)
        px = torch.full((num_cameras,), 0., device=device)
        py = torch.full((num_cameras,), 0., device=device)
        intrinsic = (fx, fy, px, py)

        # To support multiple cameras, we extend the batch size of the faces.
        face_shape = face_shape.repeat(num_cameras, 1, 1)
        face_idx = face_idx.repeat(num_cameras, 1, 1)
        face_color = face_color.repeat(num_cameras, 1, 1)

        # Rendering the image
        images = render_img(face_shape, face_idx, face_color, image_size, extrinsic, intrinsic, device=device)

        if step < 1:
            print("Num cameras:", num_cameras)
            print("Image shape:", images.shape)

        # Fake loss here.
        # loss = images.pow(2).sum()
        loss = criterion(images[..., :3])

        optim.zero_grad()
        loss.backward()
        optim.step()


    for i, image in enumerate(images.detach().cpu().numpy()):

        path_str = f"output/multi_camera_test_view{i:02}.png"
        path = os.path.split(path_str)[0]
        if os.path.exists(path) is False:
            os.makedirs(path)

        from PIL import Image
        image = np.uint8(image[:, :, :3] * 255.0)
        img = Image.fromarray(image)
        img.save(path_str)

    # face_shape = face_shape.detach().cpu().numpy()
    # face_color = face_color.detach().cpu().numpy()
    #
    # face_shape = np.squeeze(face_shape)
    # face_color = np.squeeze(face_color)
    # save_obj(path_str + '.obj', face_shape, bfm.tri, np.clip(face_color, 0, 1.0))  # 3D reconstruction face (in canonical view)


if __name__ == '__main__':
    recon()
