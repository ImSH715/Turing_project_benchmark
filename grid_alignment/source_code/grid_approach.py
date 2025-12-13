import os
import cv2
import torch
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import rowcol
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision

# -------------------------------------------------
# Config
# -------------------------------------------------
TIF_PATH = "../Dataset/Orange_trees.tif"
CSV_PATH = "../Dataset/random_point.csv"
MODEL_PATH = "model_final.pth"
OUT_DIR = "grid_outputs"
GRID_SIZE = 256
SCORE_THRESH = 0.5

os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------------------------------
# Model
# -------------------------------------------------
def load_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# -------------------------------------------------
# Extract grid
# -------------------------------------------------
def extract_grid(img, cx, cy, size):
    h, w, _ = img.shape
    half = size // 2

    x1 = max(cx - half, 0)
    y1 = max(cy - half, 0)
    x2 = min(cx + half, w)
    y2 = min(cy + half, h)

    if (x2 - x1) < size or (y2 - y1) < size:
        return None, None

    return img[y1:y2, x1:x2].copy(), (x1, y1)

# -------------------------------------------------
# Main
# -------------------------------------------------
def run():
    print("Starting grid-based detection")

    device = torch.device("cpu")
    print("Device:", device)

    model = load_model()
    print("Model loaded")

    # Load raster
    with rasterio.open(TIF_PATH) as src:
        img = src.read().transpose(1, 2, 0)
        transform = src.transform

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    print("Using CSV columns:", df.columns.tolist())

    for idx, row in df.iterrows():
        x_geo = row["long"]
        y_geo = row["lat"]

        # Geo → pixel
        py, px = rowcol(transform, x_geo, y_geo)

        if px < 0 or py < 0 or px >= img.shape[1] or py >= img.shape[0]:
            continue

        grid, (x0, y0) = extract_grid(img, px, py, GRID_SIZE)
        if grid is None:
            continue

        # Model inference
        tensor = torch.from_numpy(grid).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            outputs = model([tensor])[0]

        boxes = outputs["boxes"].numpy()
        scores = outputs["scores"].numpy()

        # Visualization
        vis = grid.copy()
        detected = False

        for box, score in zip(boxes, scores):
            if score < SCORE_THRESH:
                continue
            detected = True
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                f"{score:.2f}",
                (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        if not detected:
            cv2.putText(
                vis,
                "NO DETECTION",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2
            )

        out_path = os.path.join(OUT_DIR, f"point_{idx}.png")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    print("Done")

# -------------------------------------------------
if __name__ == "__main__":
    run()
