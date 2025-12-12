# LoG blob detection for tree crown centers
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import blob_log

# Parameters - tune for your image
IMAGE_PATH = "../Dataset/Orange_trees.tif"
MIN_SIGMA = 3     # smallest blob sigma
MAX_SIGMA = 25    # largest blob sigma
NUM_SIGMA = 10
THRESHOLD = 0.02  # lower -> more detections

# Load image and convert to grayscale (or use vegetation index)
img = np.array(Image.open(IMAGE_PATH))
gray = rgb2gray(img)  # float in [0,1]

# Detect blobs (LoG)
blobs = blob_log(gray, min_sigma=MIN_SIGMA, max_sigma=MAX_SIGMA,
                 num_sigma=NUM_SIGMA, threshold=THRESHOLD)

# blob_log returns (y, x, sigma)
centers = blobs[:, :2][:, ::-1]  # convert to (x,y) order if needed
radii = blobs[:, 2] * np.sqrt(2)

print("Detected blobs:", len(blobs))

# Visualization
fig, ax = plt.subplots(figsize=(12,8))
ax.imshow(img)
for y, x, s in blobs:
    c = plt.Circle((x, y), s*np.sqrt(2), color='red', linewidth=1.0, fill=False)
    ax.add_patch(c)
plt.title("LoG Blob Detections")
plt.axis('off')
plt.show()
