# -----------------------------------------------------------
# tiled_embedder.py  ·  Python ≥3.8  ·  Torch ≥2.2, TorchVision ≥0.17
# -----------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import (
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights,
)
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from typing import Callable, Sequence, Tuple, Optional

# ────────────────────────────────────────────────────────────
#  1.  RESNET BACKBONE (unchanged except the classifier is n-op)
# ────────────────────────────────────────────────────────────
class _ResNetBackbone(nn.Module):
    _ARCHS = {
        "resnet18":  (models.resnet18,  ResNet18_Weights.IMAGENET1K_V1),
        "resnet34":  (models.resnet34,  ResNet34_Weights.IMAGENET1K_V1),
        "resnet50":  (models.resnet50,  ResNet50_Weights.IMAGENET1K_V2),
        "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2),
        "resnet152": (models.resnet152, ResNet152_Weights.IMAGENET1K_V2),
    }

    def __init__(self, arch: str = "resnet18", device: Optional[str] = None):
        super().__init__()

        if arch.lower() not in self._ARCHS:
            raise ValueError(f"`arch` must be one of {list(self._ARCHS)}")
        ctor, weights = self._ARCHS[arch.lower()]

        self.model = ctor(weights=weights)
        self.embedding_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()
        for p in self.model.parameters():  # freeze backbone
            p.requires_grad = False

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:       # (B, 3, H, W)
        return self.model(x.to(self.device, non_blocking=True))  # (B, D)

# ────────────────────────────────────────────────────────────
#  2.  PUBLIC WRAPPER WITH TILING + K-MEANS DISTANCE
# ────────────────────────────────────────────────────────────
class ResNetTiledEmbedder:
    """
    • embed_frame(...)       → 1-row tensor whose length = tiles * D  
    • kmeans_distance(...)   → mean cosine distance to the k nearest neighbours
    """

    def __init__(self, arch: str = "resnet18", device: Optional[str] = None,
                 input_size: int = 224):
        self.backbone = _ResNetBackbone(arch, device)
        self.device   = self.backbone.device
        self.tile_dim = self.backbone.embedding_dim
        self.preprocess = transforms.Compose([
            transforms.Resize((input_size, input_size),
                              interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ── 2-a.  Create embedding from (frame, ROI, tiler) ────────────
    def embed_frame(
        self,
        frame: np.ndarray,
        roi: Tuple[float, float, float, float],
        tiler: Callable[[np.ndarray], Sequence[np.ndarray]],
    ) -> torch.Tensor:                     # → (1,  tiles * D)
        nx, ny, nw, nh = roi
        h, w = frame.shape[:2]
        x0, y0 = int(nx * w), int(ny * h)
        x1, y1 = int((nx + nw) * w), int((ny + nh) * h)
        roi_img = frame[y0:y1, x0:x1].copy()

        tiles = tiler(roi_img)
        if not tiles:
            raise ValueError("Tiler returned no tiles.")

        batch = torch.stack([
            self.preprocess(Image.fromarray(tile)) for tile in tiles
        ]).to(self.device)                               # (T, 3, H, W)

        with torch.inference_mode():
            embs = self.backbone(batch)                  # (T, D)

        return embs.view(1, -1).cpu()                    # (1, T*D)

    # ── 2-b.  Fast “k-means” distance (mean k-NN cosine distance) ──
    @staticmethod
    def kmeans_distance(
        embedding_set: torch.Tensor,  # (N, D′)  — leave on CPU
        embedding:     torch.Tensor,  # (1, D′)  — leave on CPU
        k: int = 5,
    ) -> float:
        """
        1. Cosine-normalise both inputs.
        2. Compute cosine similarities in a vectorised way.
        3. Convert to *distance* = 1 - similarity.
        4. Return the mean of the k smallest distances.
        """
        if embedding.ndim == 1:
            embedding = embedding.unsqueeze(0)

        embedding_set = F.normalize(embedding_set, dim=1)
        embedding     = F.normalize(embedding,     dim=1)

        # (1, D′) · (D′, N) → (1, N)  → flatten to (N,)
        similarities = embedding @ embedding_set.T
        distances    = 1.0 - similarities.squeeze(0)

        k = min(k, distances.numel())
        return distances.topk(k, largest=False).values.mean().item()
