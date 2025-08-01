#!/usr/bin/env python3
# anomaly_scoring.py
# ----------------------------------------------------------
# Usage:
# python anomaly_scoring.py  --good_dir D:\temp_folder\good_images --test_dir D:\temp_folder\test_images --arch resnet18 --device cuda --k 5
#
# Requirements:
#   pip install torch torchvision opencv-python matplotlib tqdm
# ----------------------------------------------------------
import argparse, glob, os, cv2, torch, numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tiled_embedder import ResNetTiledEmbedder


# ─── change or replace with your own tiler ────────────────
def simple_2x2_tiler(img: np.ndarray):
    """Split an image into 2 x 2 tiles (4 tiles total)."""
    h, w = img.shape[:2]
    h2, w2 = h // 2, w // 2
    return [
        img[0:h2, 0:w2], img[0:h2, w2:w],
        img[h2:h, 0:w2], img[h2:h, w2:w],
    ]

def simple_4x1_tiler(img: np.ndarray):
    """Split an image into 4 horizontal tiles (4 tiles total)."""
    h, w = img.shape[:2]
    h1 = h // 4
    return [
        img[0:h1, :], img[h1:2*h1, :],
        img[2*h1:3*h1, :], img[3*h1:h, :]
    ]

def load_and_embed(folder: str,
                   embedder: ResNetTiledEmbedder,
                   roi=(0.0, 0.0, 1.0, 1.0)):
    """Return (T, D) tensor of embeddings for every image in *folder*."""
    paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        for p in glob.glob(os.path.join(folder, ext))
    )
    if not paths:
        raise ValueError(f"No images found in {folder!r}")

    embs = []
    for p in tqdm(paths, desc=f"Embedding {os.path.basename(folder)}"):
        frame = cv2.imread(p)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR ▶ RGB
        emb = embedder.embed_frame(frame, roi, simple_4x1_tiler)  # (1, D)
        embs.append(emb)
    return torch.cat(embs, dim=0), paths  # (N, D)

def main():
    parser = argparse.ArgumentParser(
        description="Compute k-NN cosine distance scores and plot histogram.")
    parser.add_argument("--good_dir", help="Folder with good/reference images")
    parser.add_argument("--test_dir", help="Folder with test images to score")
    parser.add_argument("--arch",   default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50",
                                 "resnet101", "resnet152"],
                        help="ResNet backbone (default: resnet18)")
    parser.add_argument("--device", default=None,
                        choices=["cpu", "cuda", None],
                        help="'cuda', 'cpu' or omit for auto")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of nearest neighbours (default: 5)")
    args = parser.parse_args()

    # ── initialise embedder ─────────────────────────────────
    embedder = ResNetTiledEmbedder(args.arch, args.device)

    # ── build reference set from *good* images ─────────────
    reference_set, _ = load_and_embed(args.good_dir, embedder)  # (N, D)

    # ── score every image in *test* folder ─────────────────
    scores, test_paths = [], []
    embs_test, test_paths = load_and_embed(args.test_dir, embedder)
    for emb in tqdm(embs_test, desc="Scoring test samples"):
        score = ResNetTiledEmbedder.kmeans_distance(reference_set, emb, k=args.k)
        scores.append(score)

    # ── histogram ──────────────────────────────────────────
    plt.hist(scores, bins=30)
    plt.title(f"k-NN (k={args.k}) cosine-distance scores")
    plt.xlabel("distance  (1 -cosine similarity)")
    plt.ylabel("number of images")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
