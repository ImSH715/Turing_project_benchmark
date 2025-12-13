import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import rasterio
from rasterio.transform import xy
import geopandas as gpd
from shapely.geometry import box

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_PATH = "../faster_rcnn_orange.pth"
IMAGE_PATH = "../Dataset/Orange_trees.tif"
OUTPUT_SHP = "tree_clown_detections.shp"

SCORE_THRESHOLD = 0.05      # lowered threshold
IOU_THRESHOLD = 0.3

PATCH_SIZE = 1000
STRIDE = 800

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -------------------------------------------------
# Model (must match training exactly)
# -------------------------------------------------
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model

# -------------------------------------------------
# NMS
# -------------------------------------------------
def apply_nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return [], []
    boxes_t = torch.tensor(boxes, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_thresh)
    return boxes_t[keep].numpy(), scores_t[keep].numpy()

# -------------------------------------------------
# Main inference
# -------------------------------------------------
def run_inference():
    # Load model
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Load GeoTIFF
    with rasterio.open(IMAGE_PATH) as src:
        img = src.read([1, 2, 3])
        transform = src.transform
        crs = src.crs

    img = np.transpose(img, (1, 2, 0))
    img = img.astype(np.float32) / 255.0

    H, W, _ = img.shape
    print(f"Image size: {W} x {H}")

    all_boxes = []
    all_scores = []

    # Sliding window inference
    for y in range(0, H, STRIDE):
        for x in range(0, W, STRIDE):
            y2 = min(y + PATCH_SIZE, H)
            x2 = min(x + PATCH_SIZE, W)

            patch = img[y:y2, x:x2]
            if patch.shape[0] < 50 or patch.shape[1] < 50:
                continue

            tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred = model(tensor)[0]

            if len(pred["scores"]) > 0:
                print(f"Patch ({x},{y}) max score:", float(pred["scores"].max()))

            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()

            for b, s in zip(boxes, scores):
                if s >= SCORE_THRESHOLD:
                    all_boxes.append([
                        b[0] + x,
                        b[1] + y,
                        b[2] + x,
                        b[3] + y
                    ])
                    all_scores.append(float(s))

    print(f"Raw detections before NMS: {len(all_boxes)}")

    # Global NMS
    final_boxes, final_scores = apply_nms(all_boxes, all_scores, IOU_THRESHOLD)
    print(f"Detections after NMS: {len(final_boxes)}")

    # Convert pixel boxes to map polygons
    geometries = []
    scores_out = []

    for box_px, score in zip(final_boxes, final_scores):
        x1, y1, x2, y2 = box_px

        x_min, y_max = xy(transform, y1, x1)
        x_max, y_min = xy(transform, y2, x2)

        geom = box(x_min, y_min, x_max, y_max)

        geometries.append(geom)
        scores_out.append(score)

    # Save shapefile
    gdf = gpd.GeoDataFrame(
        {"score": scores_out},
        geometry=geometries,
        crs=crs
    )

    gdf.to_file(OUTPUT_SHP)
    print(f"Saved shapefile: {OUTPUT_SHP}")
    print(f"Final detection count: {len(gdf)}")

# -------------------------------------------------
if __name__ == "__main__":
    run_inference()
