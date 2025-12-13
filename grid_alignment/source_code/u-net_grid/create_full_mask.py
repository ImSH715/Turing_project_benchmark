import torch
import rasterio
import numpy as np
from train_unet import UNet

MODEL = "unet_tree_crown.pth"
IMG_PATH = "Dataset/Orange_trees.tif"
OUT_MASK = "Dataset/output/mask_full.tif"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
model.eval()

with rasterio.open(IMG_PATH) as src:
    img = src.read([1, 2, 3])
    profile = src.profile

img = np.transpose(img, (1, 2, 0)).astype(np.float32) / 255.0
H, W, _ = img.shape
mask_full = np.zeros((H, W), dtype=np.float32)

TILE = 512
STRIDE = 512

with torch.no_grad():
    for y in range(0, H - TILE + 1, STRIDE):
        for x in range(0, W - TILE + 1, STRIDE):
            tile = img[y:y+TILE, x:x+TILE]
            t = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            pred = model(t)[0, 0].cpu().numpy()
            mask_full[y:y+TILE, x:x+TILE] = np.maximum(
                mask_full[y:y+TILE, x:x+TILE], pred
            )

profile.update(count=1, dtype="float32")
with rasterio.open(OUT_MASK, "w", **profile) as dst:
    dst.write(mask_full, 1)

print("Saved mask:", OUT_MASK)
