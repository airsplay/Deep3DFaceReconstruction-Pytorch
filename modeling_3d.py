import torch
import math
import numpy as np
from utils import LeastSquares


def split_coeff(coeff):
    # input: coeff with shape [1,257]
    id_coeff = coeff[:, :80]  # identity(shape) coeff of dim 80
    ex_coeff = coeff[:, 80:144]  # expression coeff of dim 64
    tex_coeff = coeff[:, 144:224]  # texture(albedo) coeff of dim 80
    angles = coeff[:, 224:227]  # Euler angles(x,y,z) for rotation of dim 3

    # lighting coeff for 3 channel SH function of dim 27
    gamma = coeff[:, 227:254]
    translation = coeff[:, 254:]  # translation coeff of dim 3

    return id_coeff, ex_coeff, tex_coeff, angles, gamma, translation


class _need_const:
    a0 = np.pi
    a1 = 2 * np.pi / np.sqrt(3.0)
    a2 = 2 * np.pi / np.sqrt(8.0)
    c0 = 1 / np.sqrt(4 * np.pi)
    c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
    c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
    d0 = 0.5 / np.sqrt(3.0)

    illu_consts = [a0, a1, a2, c0, c1, c2, d0]

    origin_size = 300
    target_size = 224
    camera_pos = 10.0


def shape_formation(id_coeff, ex_coeff, facemodel):
    # compute face shape with identity and expression coeff, based on BFM model
    # input: id_coeff with shape [1,80]
    #         ex_coeff with shape [1,64]
    # output: face_shape with shape [1,N,3], N is number of vertices

    '''
        S = mean_shape + \alpha * B_id + \beta * B_exp
    '''
    n_b = id_coeff.size(0)
    face_shape = torch.einsum('ij,aj->ai', facemodel.idBase, id_coeff) + \
        torch.einsum('ij,aj->ai', facemodel.exBase, ex_coeff) + \
        facemodel.meanshape

    face_shape = face_shape.view(n_b, -1, 3)
    # re-center face shape
    face_shape = face_shape - \
        facemodel.meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)

    return face_shape


def texture_formation(tex_coeff, facemodel):
    # compute vertex texture(albedo) with tex_coeff
    # input: tex_coeff with shape [1,N,3]
    # output: face_texture with shape [1,N,3], RGB order, range from 0-255

    '''
        T = mean_texture + \gamma * B_texture
    '''

    n_b = tex_coeff.size(0)
    face_texture = torch.einsum(
        'ij,aj->ai', facemodel.texBase, tex_coeff) + facemodel.meantex

    face_texture = face_texture.view(n_b, -1, 3)
    return face_texture


def compute_norm(face_shape, facemodel):
    # compute vertex normal using one-ring neighborhood (8 points)
    # input: face_shape with shape [1,N,3]
    # output: v_norm with shape [1,N,3]
    # https://fredriksalomonsson.files.wordpress.com/2010/10/mesh-data-structuresv2.pdf

    # vertex index for each triangle face, with shape [F,3], F is number of faces
    face_id = facemodel.tri - 1
    # adjacent face index for each vertex, with shape [N,8], N is number of vertex
    point_id = facemodel.point_buf - 1
    shape = face_shape
    v1 = shape[:, face_id[:, 0], :]
    v2 = shape[:, face_id[:, 1], :]
    v3 = shape[:, face_id[:, 2], :]
    e1 = v1 - v2
    e2 = v2 - v3
    face_norm = e1.cross(e2)  # compute normal for each face

    # normalized face_norm first
    face_norm = torch.nn.functional.normalize(face_norm, p=2, dim=2)
    empty = torch.zeros((face_norm.size(0), 1, 3),
                        dtype=face_norm.dtype, device=face_norm.device)

    # concat face_normal with a zero vector at the end
    face_norm = torch.cat((face_norm, empty), 1)

    # compute vertex normal using one-ring neighborhood
    v_norm = face_norm[:, point_id, :].sum(dim=2)
    v_norm = torch.nn.functional.normalize(v_norm, p=2, dim=2)  # normalize normal vectors
    return v_norm


def compute_rotation_matrix(angles):
    # compute rotation matrix based on 3 Euler angles
    # input: angles with shape [1,3]
    # output: rotation matrix with shape [1,3,3]
    n_b = angles.size(0)

    # https://www.cnblogs.com/larry-xia/p/11926121.html
    device = angles.device
    # compute rotation matrix for X-axis, Y-axis, Z-axis respectively
    rotation_X = torch.cat(
        [
            torch.ones([n_b, 1]).to(device),
            torch.zeros([n_b, 3]).to(device),
            torch.reshape(torch.cos(angles[:, 0]), [n_b, 1]),
            - torch.reshape(torch.sin(angles[:, 0]), [n_b, 1]),
            torch.zeros([n_b, 1]).to(device),
            torch.reshape(torch.sin(angles[:, 0]), [n_b, 1]),
            torch.reshape(torch.cos(angles[:, 0]), [n_b, 1])
        ],
        axis=1
    )
    rotation_Y = torch.cat(
        [
            torch.reshape(torch.cos(angles[:, 1]), [n_b, 1]),
            torch.zeros([n_b, 1]).to(device),
            torch.reshape(torch.sin(angles[:, 1]), [n_b, 1]),
            torch.zeros([n_b, 1]).to(device),
            torch.ones([n_b, 1]).to(device),
            torch.zeros([n_b, 1]).to(device),
            - torch.reshape(torch.sin(angles[:, 1]), [n_b, 1]),
            torch.zeros([n_b, 1]).to(device),
            torch.reshape(torch.cos(angles[:, 1]), [n_b, 1]),
        ],
        axis=1
    )
    rotation_Z = torch.cat(
        [
            torch.reshape(torch.cos(angles[:, 2]), [n_b, 1]),
            - torch.reshape(torch.sin(angles[:, 2]), [n_b, 1]),
            torch.zeros([n_b, 1]).to(device),
            torch.reshape(torch.sin(angles[:, 2]), [n_b, 1]),
            torch.reshape(torch.cos(angles[:, 2]), [n_b, 1]),
            torch.zeros([n_b, 3]).to(device),
            torch.ones([n_b, 1]).to(device),
        ],
        axis=1
    )

    rotation_X = rotation_X.reshape([n_b, 3, 3])
    rotation_Y = rotation_Y.reshape([n_b, 3, 3])
    rotation_Z = rotation_Z.reshape([n_b, 3, 3])

    # R = Rz*Ry*Rx
    rotation = rotation_Z.bmm(rotation_Y).bmm(rotation_X)

    # because our face shape is N*3, so compute the transpose of R, so that rotation shapes can be calculated as face_shape*R
    rotation = rotation.permute(0, 2, 1)

    return rotation



def illumination_layer(face_texture, norm, gamma):
    # CHJ: It's different from what I knew.
    # compute vertex color using face_texture and SH function lighting approximation
    # input: face_texture with shape [1,N,3]
    #          norm with shape [1,N,3]
    #         gamma with shape [1,27]
    # output: face_color with shape [1,N,3], RGB order, range from 0-255
    #          lighting with shape [1,N,3], color under uniform texture

    n_b, num_vertex, _ = face_texture.size()
    n_v_full = n_b * num_vertex
    gamma = gamma.view(-1, 3, 9).clone()
    gamma[:, :, 0] += 0.8

    gamma = gamma.permute(0, 2, 1)

    a0, a1, a2, c0, c1, c2, d0 = _need_const.illu_consts

    Y0 = torch.ones(n_v_full).float() * a0*c0
    if gamma.is_cuda:
        Y0 = Y0.cuda()
    norm = norm.view(-1, 3)
    nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
    arrH = []

    arrH.append(Y0)
    arrH.append(-a1*c1*ny)
    arrH.append(a1*c1*nz)
    arrH.append(-a1*c1*nx)
    arrH.append(a2*c2*nx*ny)
    arrH.append(-a2*c2*ny*nz)
    arrH.append(a2*c2*d0*(3*nz.pow(2)-1))
    arrH.append(-a2*c2*nx*nz)
    arrH.append(a2*c2*0.5*(nx.pow(2)-ny.pow(2)))

    H = torch.stack(arrH, 1)
    Y = H.view(n_b, num_vertex, 9)

    # Y shape:[batch,N,9].
    # shape:[batch,N,3]
    lighting = Y.bmm(gamma)

    face_color = face_texture * lighting

    return face_color, lighting


def rigid_transform(face_shape, rotation, translation):
    n_b = face_shape.shape[0]
    face_shape_r = face_shape.bmm(rotation)  # R has been transposed
    face_shape_t = face_shape_r + translation.view(n_b, 1, 3)
    return face_shape_t


def reconstruction(coeff, facemodel):
    # The image size is 224 * 224
    # face reconstruction with coeff and BFM model
    id_coeff, ex_coeff, tex_coeff, angles, gamma, translation = split_coeff(coeff)

    # compute face shape
    face_shape = shape_formation(id_coeff, ex_coeff, facemodel)
    # compute vertex texture(albedo)
    face_texture = texture_formation(tex_coeff, facemodel)

    # vertex normal
    face_norm = compute_norm(face_shape, facemodel)

    # rotation matrix
    rotation = compute_rotation_matrix(angles)
    face_norm_r = face_norm.bmm(rotation)

    # do rigid transformation for face shape using predicted rotation and translation
    # face_shape_t = rigid_transform(face_shape, rotation, translation)

    # compute vertex color using SH function lighting approximation
    face_color, lighting = illumination_layer(face_texture, face_norm_r, gamma)

    return face_shape, face_texture, face_color, angles, translation, gamma
