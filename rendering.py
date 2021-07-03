from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.textures import TexturesVertex
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    BlendParams
)
import torch


def render_img(face_shape, face_color, facemodel, image_size=224, extrinsic=None, intrinsic=None, device='cuda:0'):
    '''
        ref: https://github.com/facebookresearch/pytorch3d/issues/184
        The rendering function (just for test)
        Input:
            face_shape:  Tensor[1, 35709, 3]
            face_color: Tensor[1, 35709, 3] in [0, 1]
            facemodel: contains `tri` (triangles[70789, 3], index start from 1)
            extrinsic: (R, T); R float tensor of [B, 3, 3]; T float tensor of [B, 3]
    '''
    batch_size = face_shape.shape[0]

    face_color = TexturesVertex(verts_features=face_color.to(device))
    face_buf = torch.from_numpy(facemodel.tri - 1)  # index start from 1
    face_idx = face_buf.unsqueeze(0).repeat(batch_size, 1, 1)

    meshes = Meshes(face_shape.to(device), face_idx.to(device), face_color)

    #R = torch.eye(3).view(1, 3, 3).to(device)
    #R[0, 0, 0] *= -1.0
    #T = torch.zeros([1, 3]).to(device)

    R, T = extrinsic
    fx, fy, px, py = intrinsic

    half_size = (image_size - 1.0) / 2
    focal_length = torch.tensor([fx / half_size, fy / half_size], dtype=torch.float32, device=device).reshape(1, 2)
    #principal_point = torch.tensor( [(half_size - px) / half_size, (py - half_size) / half_size], dtype=torch.float32, device=device).reshape(1, 2)
    principal_point = torch.tensor([px / half_size, py / half_size], dtype=torch.float32, device=device).reshape(1, 2)

    cameras = PerspectiveCameras(
        device=device,
        R=R,
        T=T,
        focal_length=focal_length,
        principal_point=principal_point
    )

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1
    )

    lights = PointLights(
        device=device,
        ambient_color=((1.0, 1.0, 1.0),),
        diffuse_color=((0.0, 0.0, 0.0),),
        specular_color=((0.0, 0.0, 0.0),),
        location=((0.0, 0.0, 1e5),)
    )

    blend_params = BlendParams(background_color=(0.0, 0.0, 0.0))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            blend_params=blend_params
        )
    )
    images = renderer(meshes)

    # TODO: There would be no gradient if clamp works.
    images = torch.clamp(images, 0.0, 1.0)

    return images
