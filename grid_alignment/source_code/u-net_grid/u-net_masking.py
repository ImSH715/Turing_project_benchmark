import os
import numpy as np
import rasterio
import cv2

IMG_PATH = "Dataset/Orange_trees.tif"
OUT_IMG_DIR = "Dataset/tiles/images"
OUT_MASK_DIR = "Dataset/tiles/masks"

TILE_SIZE = 512
STRIDE = 512

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_MASK_DIR, exist_ok=True)

def vegetation_mask(img):
    R = img[:, :, 0].astype(np.int16)
    G = img[:, :, 1].astype(np.int16)
    B = img[:, :, 2].astype(np.int16)
    idx = 2 * G - R - B
    mask = (idx > 20) & (G > 40)
    mask = mask.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

with rasterio.open(IMG_PATH) as src:
    img = src.read([1, 2, 3])
    img = np.transpose(img, (1, 2, 0))

H, W, _ = img.shape
idx = 0

for y in range(0, H - TILE_SIZE + 1, STRIDE):
    for x in range(0, W - TILE_SIZE + 1, STRIDE):
        tile = img[y:y+TILE_SIZE, x:x+TILE_SIZE]
        mask = vegetation_mask(tile)

        cv2.imwrite(f"{OUT_IMG_DIR}/img_{idx:04d}.tif", tile)
        cv2.imwrite(f"{OUT_MASK_DIR}/mask_{idx:04d}.tif", mask)
        idx += 1

print(f"Generated {idx} tiles")
