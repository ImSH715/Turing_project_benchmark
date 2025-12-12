import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# Load Pretrained U-Net
model = smp.Unet(
    encoder_name="resnet34",       # imagenet pretrained
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,                     # binary segmentation
    activation="sigmoid"
)
model.eval()

# Load and preprocess image
img = np.array(Image.open("../Dataset/Orange_trees.tif"))
orig_h, orig_w = img.shape[:2]

# Resize for U-Net input (512×512)
inp = cv2.resize(img, (512, 512))
inp = inp.astype(np.float32) / 255.0
inp = np.transpose(inp, (2, 0, 1))        # HWC → CHW
inp = torch.tensor(inp).unsqueeze(0)

# U-Net prediction
with torch.no_grad():
    pred = model(inp)             # (1,1,512,512)
mask = pred.squeeze().numpy()

# Resize mask back to original size
mask = cv2.resize(mask, (orig_w, orig_h))

# Threshold
binary_mask = (mask > 0.5).astype(np.uint8)

# Extract crown regions and centers
num_labels, labels = cv2.connectedComponents(binary_mask)

centers = []
for lbl in range(1, num_labels):  # 0 = background
    ys, xs = np.where(labels == lbl)
    cx, cy = xs.mean(), ys.mean()
    centers.append((cx, cy))
centers = np.array(centers)

# Visualization
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.imshow(binary_mask, cmap="jet", alpha=0.4)
plt.scatter(centers[:,0], centers[:,1], s=20, c="yellow")
plt.title("U-Net Crown Segmentation + Centers")
plt.axis("off")
plt.show()
