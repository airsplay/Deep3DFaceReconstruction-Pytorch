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


def render_img(face_shape, face_idx, face_color, image_size=224, extrinsic=None, intrinsic=None, device='cuda:0'):
    '''
        ref: https://github.com/facebookresearch/pytorch3d/issues/184
        We support the batching over faces and cameras. Here are the three cases:
            1. #face = 1 and #camera > 1: applying multiple camera to the same face.
            2. #face > 1 and #camera = 1: applying the same camear to multiple face.
            3. #face > 1 and #camera = #face: applying each camera[i] to face[i]
        Input:
            face_shape:  Tensor[#face, 35709, 3]
            face_idx: Tensor[#face, 70789, 3]
            face_color: Tensor[#face, 35709p, 3] in [0, 1]
            image_size: int. We do not support batching here to allow smooth downstream optimizations.
            extrinsic: (R, T); R float tensor of [#camera, 3, 3]; T float tensor of [#camera, 3]
            intrinsic: (fx, fy, px, py): each is a tensor of shape [#camera,].
                       All use screen coordinates.
        Returns:
            images: [B, image_size, image_size, 4] RGBD, ranging from 0 to 1
    '''
    face_color = TexturesVertex(verts_features=face_color.to(device))
    meshes = Meshes(face_shape.to(device), face_idx.to(device), face_color)

    R, T = extrinsic                # R: [n, 3, 3], T: [n, 3]
    fx, fy, px, py = intrinsic      # fx, fy, px, py: torch.tensor[n]

    half_size = (image_size - 1.0) / 2
    focal_length = torch.stack([fx / half_size, fy / half_size], 1)         # (b,) + (b,) --> (b, 2)
    principal_point = torch.stack([px / half_size, py / half_size], 1)      # (b,) + (b,) --> (b, 2)

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
