import sys
sys.path.append(".")

import argparse
import os
import math

import numpy as np
import torch
from torch.optim import Adam
import tqdm

from bfmface.load_data import transfer_BFM09, BFM
from bfmface.modeling_3d import reconstruction, compute_rotation_matrix
from bfmface.loss import ClipLoss
from bfmface.rendering import render_img
from bfmface.html_converter import dump_html


def recon():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", default="an angry man | real human")
    parser.add_argument("--optim-steps", type=int, default=1000)
    args = parser.parse_args()

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
        exp_coef,
        texture_coef,
    ]           # We only optimize the shape, expression, and texture coefficients.
    for coef in optimized_coefs:
        coef.requires_grad = True
    optim = Adam(optimized_coefs, lr=5e-3)

    # Loss and regularizations
    criterion = ClipLoss(prompt=args.prompt).to(device)
    reg_weight = 0.1
    # regularizer = L2Regularizer(optimized_coefs)

    num_steps = args.optim_steps
    image_size = 224
    print("Use prompt:", args.prompt)
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
        #   We capture the face with *multiple cameras* so that the images are consistent across different views.
        #   Please make sure that psi_samples and theta_sampled are odd numbers so that the centering camera is used.
        psi_range = math.pi / 6
        psi_samples = 5
        theta_range = math.pi / 6
        theta_samples = 5
        psi = torch.linspace(-psi_range, psi_range, psi_samples, device=device)     # The samples of psi angles
        theta = torch.linspace(-theta_range, theta_range, theta_samples, device=device) # The samples of theta angles.
        num_cameras = psi_samples * theta_samples
        phi = torch.zeros(1, device=device)
        euler = torch.cartesian_prod(psi, theta, phi)
        # camera_direction = compute_rotation_matrix(euler)
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
        #   Note: to support CLIP, we change the image_size from 300 to 224.
        #         Thus the focus length in 3D space would be different (1015 / 224 instead of 1015 / 300)
        focus_length_in_screen = 1015.0
        fx = torch.full((num_cameras,), focus_length_in_screen, device=device)
        fy = torch.full((num_cameras,), focus_length_in_screen, device=device)
        px = torch.full((num_cameras,), 0., device=device)
        py = torch.full((num_cameras,), 0., device=device)
        intrinsic = (fx, fy, px, py)

        # To support multiple cameras, we expand the same face to the number of cameras.
        face_shape = face_shape.repeat(num_cameras, 1, 1)
        face_idx = face_idx.repeat(num_cameras, 1, 1)
        face_color = face_color.repeat(num_cameras, 1, 1)

        # Rendering the image
        images = render_img(face_shape, face_idx, face_color, image_size, extrinsic, intrinsic, device=device)

        if step < 1:
            print("Num cameras:", num_cameras)
            print("Image shape:", images.shape)

        # The loss compute the matching score between the images and the language prompt
        loss = criterion(images[..., :3])

        optim.zero_grad()
        loss.backward()
        optim.step()

    # Save images
    img_paths = []
    for i, image in enumerate(images.detach().cpu().numpy()):
        path_str = f"output/multi_camera_test_view{i:02}.png"
        img_paths.append(path_str)

        path_dir = os.path.split(path_str)[0]
        if os.path.exists(path_dir) is False:
            os.makedirs(path_dir)

        from PIL import Image
        image = np.uint8(image[:, :, :3] * 255.0)
        img = Image.fromarray(image)
        img.save(path_str)

    # It is available at https://www.cs.unc.edu/~airsplay/results.html
    dump_html(img_paths, "results.html")

    # face_shape = face_shape.detach().cpu().numpy()
    # face_color = face_color.detach().cpu().numpy()
    #
    # face_shape = np.squeeze(face_shape)
    # face_color = np.squeeze(face_color)
    # save_obj(path_str + '.obj', face_shape, bfm.tri, np.clip(face_color, 0, 1.0))  # 3D reconstruction face (in canonical view)


if __name__ == '__main__':
    recon()
