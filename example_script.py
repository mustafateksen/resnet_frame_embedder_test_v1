import cv2
import torch
from tiled_embedder import ResNetTiledEmbedder

# 1 . initialise
embedder = ResNetTiledEmbedder("resnet18", device="cuda")  # or "cpu"

# 2 . supply a tiler (replace with your own)
def simple_2x2_tiler(img):
    h, w = img.shape[:2]
    h2, w2 = h // 2, w // 2
    return [
        img[0:h2, 0:w2], img[0:h2, w2:w],
        img[h2:h, 0:w2], img[h2:h, w2:w],
    ]

# 3 . embed a frame
frame = cv2.imread("example_image.bmp")[:, :, ::-1]  # BGR → RGB
roi   = (0.0, 0.0, 1.0, 1.0)                  # whole image
vec   = embedder.embed_frame(frame, roi, simple_2x2_tiler)  # (1, 4*2048)

# 4 . compute “k-means” distance
reference_set = torch.randn(100, vec.shape[1])  # dummy embeddings

score = ResNetTiledEmbedder.kmeans_distance(reference_set, vec, k=5)
print("distance:", score)
