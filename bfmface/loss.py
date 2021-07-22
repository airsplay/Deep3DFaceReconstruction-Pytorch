import clip
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torchvision.transforms import functional as TF
from PIL import Image


def fetch(url_or_path):
    import requests
    import io
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')


def resize_image(image, out_size):
    ratio = image.size[0] / image.size[1]
    area = min(image.size[0] * image.size[1], out_size[0] * out_size[1])
    size = round((area * ratio)**0.5), round((area / ratio)**0.5)
    return image.resize(size, Image.LANCZOS)


class ClipLoss(nn.Module):
    def __init__(self, clip_model="ViT-B/32", prompt="human face.",
                 target_image_paths=None):
        super(ClipLoss, self).__init__()
        self.clip_model = clip_model
        self.device = "cuda"

        if type(prompt) is str:
            # self.prompt = [prompt]
            self.prompt = prompt.split("|")
        else:
            self.prompt = prompt
        if len(self.prompt) == 0:
            self.prompt = None

        if type(target_image_paths) is str:
            # self.prompt = [prompt]
            self.target_image_paths = target_image_paths.split("|")
        else:
            self.target_image_paths = target_image_paths
        # if len(self.target_image_paths) == 0:
        #     self.target_image_paths = None

        self.model = None

    def to(self, device):
        self.device = device
        return super(ClipLoss, self).to(device)

    def lazy_loading(self):
        # Load the clip model
        self.model = clip.load(self.clip_model, self.device, jit=False)[0].eval().requires_grad_(False)
        self.image_normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        # Convert text to features
        if self.prompt is not None:
            text = clip.tokenize(self.prompt).to(self.device)
            text_features = self.model.encode_text(text)
            self.text_features = F.normalize(text_features, dim=-1)

        if self.target_image_paths is not None:
            sideX, sideY = 224, 224
            target_image_features = []
            for target_image_path in self.target_image_paths:
                img = resize_image(Image.open(fetch(target_image_path)).convert('RGB'), (sideX, sideY))
                batch = TF.to_tensor(img).unsqueeze(0).to(self.device)
                target_image_features.append(self.model.encode_image(self.image_normalize(batch)).float())
            target_image_features = torch.cat(target_image_features, 0)
            self.target_image_features = F.normalize(target_image_features)

    def forward(self, img):
        """
        Args:
            img: Tensor of shape [batch, N, M, 3]
        Returns:
            a scalar loss
        """
        if self.model is None:
            self.lazy_loading()

        # Convert image to features
        img = img.movedim(3, 1)            # (b, N, M, 3) --> (b, 3, N, M)
        img = self.image_normalize(img)
        image_features = self.model.encode_image(img)           # (b, dim)
        image_features = F.normalize(image_features, dim=-1)    # (b, dim)

        # Copy from VQGAN+CLIP (z+quantize method).ipynb
        dists = (image_features.unsqueeze(0) - self.text_features.unsqueeze(1)).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
        return dists.mean()

    def extra_repr(self) -> str:
        return "Text prompt: " + str(self.prompt) + ".  Target Image: " + str(self.target_image_paths)


# class L2Regularizer(nn.Module):
#     def forward(self, optim_params, l2_weight):
#         optim_params_tensor: torch.tensor = torch.cat(optim_params.flatten())
#         return optim_params_tensor ** 2).sum()


