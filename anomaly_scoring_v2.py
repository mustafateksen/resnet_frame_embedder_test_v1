#!/usr/bin/env python3
# anomaly_scoring.py
# ----------------------------------------------------------
# Example:
# python anomaly_scoring_v2.py --good_dir "D:/temp_folder/good_images" --test_dir "D:/temp_folder/test_images" --anomaly_labeled_folder "D:/temp_folder/seperated_images" --threshold 0.20 --arch resnet18 --device cuda --k 5


# ----------------------------------------------------------
import argparse, glob, os, cv2, torch, numpy as np, shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from tiled_embedder import ResNetTiledEmbedder


# ─── tiler (replace if you like) ───────────────────────────
def simple_4x1_tiler(img: np.ndarray):
    h, w = img.shape[:2]
    h1 = h // 4
    return [
        img[0:h1,     :], img[h1:2*h1, :],
        img[2*h1:3*h1,:], img[3*h1:h,  :]
    ]

def load_and_embed(folder: str, embedder: ResNetTiledEmbedder,
                   roi=(0.275, 0.0, (0.706-0.275), 1.0)):
    paths = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        for p in glob.glob(os.path.join(folder, ext))
    )
    if not paths:
        raise ValueError(f"No images found in {folder!r}")

    embs = []
    for p in tqdm(paths, desc=f"Embedding {os.path.basename(folder)}"):
        frame = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        embs.append(embedder.embed_frame(frame, roi, simple_4x1_tiler))
    return torch.cat(embs, dim=0), paths


def main():
    ap = argparse.ArgumentParser(
        description="Compute k-NN cosine-distance scores, copy anomalies, "
                    "and plot a histogram.")
    ap.add_argument("--good_dir", required=True)
    ap.add_argument("--test_dir", required=True)
    ap.add_argument("--anomaly_labeled_folder", required=True,
                    help="Where anomalous images (score > threshold) are copied")
    ap.add_argument("--threshold", type=float, required=True,
                    help="Manual anomaly threshold")
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "resnet34", "resnet50",
                            "resnet101", "resnet152"])
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", None])
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    # ── prepare folders ─────────────────────────────────────
    anomaly_dir = Path(args.anomaly_labeled_folder)
    anomaly_dir.mkdir(parents=True, exist_ok=True)

    # ── initialise embedder ─────────────────────────────────
    embedder = ResNetTiledEmbedder(args.arch, args.device)

    # ── reference set (good images) ─────────────────────────
    reference_set, _ = load_and_embed(args.good_dir, embedder)

    # ── score test images ───────────────────────────────────
    MAX_ITEM_COUNT = 9999
    scores, test_paths = [], []
    embs_test, test_paths = load_and_embed(args.test_dir, embedder)
    embs_test = embs_test[:MAX_ITEM_COUNT]  # limit to first N items
    test_paths = test_paths[:MAX_ITEM_COUNT]

    for emb, p in tqdm(zip(embs_test, test_paths),
                    total=len(test_paths),
                    desc="Scoring"):
        score = ResNetTiledEmbedder.kmeans_distance(reference_set, emb, k=args.k)
        scores.append(score)

    # ── sort once, then copy in ranked order ──────────────────
    ranked = sorted(zip(scores, test_paths), key=lambda x: x[0], reverse=True)

    for rank, (score, src_path) in enumerate(ranked, start=1):
        if score <= args.threshold:
            break                                       # remaining scores are lower
        # build a name that sorts naturally: 00001_0.873421_original.jpg
        name = f"{rank:05d}_{score:.6f}_{os.path.basename(src_path)}"
        shutil.copy2(src_path, anomaly_dir / name)

    print(f"\n→ {rank - 1} images (score > {args.threshold}) saved to '{anomaly_dir}'.")

    # ── histogram ──────────────────────────────────────────
    plt.hist(scores, bins=30)
    plt.axvline(args.threshold, linestyle="--")
    plt.title(f"k-NN (k={args.k}) cosine-distance scores")
    plt.xlabel("distance  (1 - cos sim)")
    plt.ylabel("number of images")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
