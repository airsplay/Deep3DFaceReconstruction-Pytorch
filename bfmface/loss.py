import clip
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import Normalize


class ClipLoss(nn.Module):
    def __init__(self, clip_model="ViT-B/32", prompt="human face."):
        super(ClipLoss, self).__init__()
        self.clip_model = clip_model
        self.device = "cuda"
        if type(prompt) is str:
            self.prompt = [prompt]
        else:
            self.prompt = prompt
        self.model = None

    def to(self, device):
        self.device = device
        return super(ClipLoss, self).to(device)

    def lazy_loading(self):
        # Load the clip model
        self.model = clip.load(self.clip_model, self.device, jit=False)[0].eval().requires_grad_(False)
        self.image_normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        # Convert text to features
        text = clip.tokenize(self.prompt).to(self.device)
        text_features = self.model.encode_text(text)
        self.text_features = F.normalize(text_features, dim=-1)

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


class L2Regularizer(nn.Module):
    def forward(self, optimized_coef):
        torch.cat(optimized_coef)


