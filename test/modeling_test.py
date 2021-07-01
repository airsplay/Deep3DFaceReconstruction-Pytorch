import sys
sys.path.append(".")
print(sys.path)

import os
import glob
import torch
import numpy as np
from models.resnet_50 import resnet50_use
from load_data import transfer_BFM09, BFM, load_img, Preprocess, save_obj
from reconstruction_mesh import reconstruction, render_img, transform_face_shape, estimate_intrinsic



def recon():
    # read BFM face model
    # transfer original BFM model to our model
    if not os.path.isfile('BFM/BFM_model_front.mat'):
        transfer_BFM09()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    bfm = BFM(r'BFM/BFM_model_front.mat', device)

    # read standard landmarks for preprocessing images
    lm3D = bfm.load_lm3d()

    arr_coef = model(input_img)

    coef = torch.cat(arr_coef, 1)

    # reconstruct 3D face with output coefficients and face model
    face_shape, face_texture, face_color, landmarks_2d, z_buffer, angles, translation, gamma = reconstruction(coef, bfm)

    fx, px, fy, py = estimate_intrinsic(landmarks_2d, transform_params, z_buffer, face_shape, bfm, angles, translation)

    face_shape_t = transform_face_shape(face_shape, angles, translation)
    face_color = face_color / 255.0
    face_shape_t[:, :, 2] = 10.0 - face_shape_t[:, :, 2]

    images = render_img(face_shape_t, face_color, bfm, 300, fx, fy, px, py)
    images = images.detach().cpu().numpy()
    images = np.squeeze(images)

    path_str = file.replace(image_path, save_path)
    path = os.path.split(path_str)[0]
    if os.path.exists(path) is False:
        os.makedirs(path)

    from PIL import Image
    images = np.uint8(images[:, :, :3] * 255.0)
    # init_img = np.array(img)
    # init_img[images != 0] = 0
    # images += init_img
    img = Image.fromarray(images)
    img.save(file.replace(image_path, save_path).replace('jpg', 'png'))

    face_shape = face_shape.detach().cpu().numpy()
    face_color = face_color.detach().cpu().numpy()

    face_shape = np.squeeze(face_shape)
    face_color = np.squeeze(face_color)
    save_obj(file.replace(image_path, save_path).replace('.jpg', '_mesh.obj'), face_shape, bfm.tri, np.clip(face_color, 0, 1.0))  # 3D reconstruction face (in canonical view)

if __name__ == '__main__':
    recon()
