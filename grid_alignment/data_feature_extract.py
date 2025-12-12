import torch
from torchvision import models, transforms
import torch.nn.functional as F
import tifffile as tiff
import numpy as np
import cv2
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: ",device)
# Loading image
img = tiff.imread("Dataset/Orange_trees.tif")# shape: H x W x C

# Not the RGB but finding channel
if img.ndim == 2:  # Grayscale
    img = img[..., None].repeat(3, axis=2)
elif img.shape[2] > 3:  # Use upper 3 channel, if 3>channel
    img = img[:, :, :3]

# Convert to float32 and normalise
img = img.astype(np.float32) / 255.0
# Convert to torch tensor : [batch, channel, height, width]
img_t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)
H, W = img_t.shape[2], img_t.shape[3]
print("Image tensor shape:", img.shape)

# Pretrained ResNet-18
model = models.resnet18(pretrained=True).to(device)
model.eval()
# Delete last layer, and extract feature only
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1]).to(device)

### Feature Extraction
with torch.no_grad():
    reference_feature = feature_extractor(img_t)
    reference_feature = reference_feature.view(1,-1)
np.save("Orange_trees_features.npy", reference_feature.cpu().numpy())

### Feature Detection
# Patch parameters
patch_size = 224
stride = 112

detections = []

with torch.no_grad():
    for y in range(0, H-patch_size+1, stride):
        for x in range(0, W-patch_size+1, stride):
            patch = img_t[:, :, y:y+patch_size, x:x+patch_size]
            feat = feature_extractor(patch)
            feat = feat.view(1,-1)
            cos_sim = F.cosine_similarity(feat, reference_feature)
            if cos_sim.item() > 0.8:
                detections.append((x, y, cos_sim.item()))

print("Detected patches:", len(detections))

### Non-maximum suppression (NMS)
def nms(boxes, scores, iou_threshold=0.3):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    areas = (x2 - x1 + 1)*(y2 - y1 +1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 +1)
        h = np.maximum(0.0, yy2 - yy1 +1)
        inter = w*h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds +1]
    return keep

boxes = [(x,y,patch_size,patch_size) for (x,y,sim) in detections]
scores = [sim for (x,y,sim) in detections]
keep_idx = nms(boxes, scores, iou_threshold=0.3)
nms_detections = [detections[i] for i in keep_idx]
print("Detections after NMS:", len(nms_detections))

### Visualisation
output = (img * 255).astype(np.uint8).copy()
# Drawing detected box
for (x, y, sim) in nms_detections:
    color_val = int((sim-0.8)/0.2*255)
    color = (255-color_val,0,color_val)  
    cv2.rectangle(output, (x,y), (x+patch_size, y+patch_size), color, 2)
# Saving img
cv2.imwrite("Detected_image.png", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
plt.imshow(output)
plt.axis("off")
plt.show()