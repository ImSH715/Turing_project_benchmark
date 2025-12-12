import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import tifffile as tiff
import numpy as np
import cv2
import os
import pandas as pd

# --- Configuration ---
# Path to the trained model weights
MODEL_PATH = "faster_rcnn_orange.pth"
# Path to the target image
IMAGE_PATH = "Dataset/Orange_trees.tif"
# Path to save the visualization result
OUTPUT_IMAGE_PATH = "Result_Detection.png"
# Path to save the detected coordinates (Pixel coordinates)
OUTPUT_CSV_PATH = "Result_Detection.csv"

# Thresholds
SCORE_THRESHOLD = 0.5       # Confidence score threshold (0.0 to 1.0)
IOU_THRESHOLD = 0.2         # IoU threshold for NMS (removing overlapping boxes)

# Patch settings for sliding window (Crucial for large satellite images)
PATCH_SIZE = 1000           # Size of the crop to feed into the model
STRIDE = 800                # Step size (Overlap = PATCH_SIZE - STRIDE)

# Device setup
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# --- 1. Model Definition (Must match training architecture) ---
def get_model(num_classes):
    # Load pre-trained Faster R-CNN ResNet50
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None # We load our own weights, so no need to download pre-trained weights
    )
    # Replace the head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- 2. Helper Functions ---
def apply_nms_globally(all_boxes, all_scores, iou_thresh=0.3):
    """
    Applies Non-Maximum Suppression (NMS) to the merged list of boxes from all patches.
    This removes duplicate detections in overlapping regions.
    """
    if len(all_boxes) == 0:
        return [], []
    
    # Convert to tensors
    boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    
    # Apply NMS
    keep_indices = torchvision.ops.nms(boxes_t, scores_t, iou_thresh)
    
    final_boxes = boxes_t[keep_indices].numpy()
    final_scores = scores_t[keep_indices].numpy()
    
    return final_boxes, final_scores

# --- 3. Main Detection Logic ---
def detect_trees():
    # A. Load Model
    num_classes = 2 # 1(Tree) + 1(Background)
    model = get_model(num_classes)
    
    if os.path.exists(MODEL_PATH):
        try:
            # Load weights (handle potential GPU/CPU mismatch)
            state_dict = torch.load(MODEL_PATH, map_location=device)
            model.load_state_dict(state_dict)
            print(f"[Info] Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"[Error] Failed to load model weights: {e}")
            return
    else:
        print(f"[Error] Model file not found at {MODEL_PATH}. Please train the model first.")
        return

    model.to(device)
    model.eval() # Set to evaluation mode

    # B. Load Image
    if not os.path.exists(IMAGE_PATH):
        print(f"[Error] Image file not found at {IMAGE_PATH}")
        return

    print(f"[Info] Loading image from {IMAGE_PATH}...")
    img_full = tiff.imread(IMAGE_PATH)
    
    # Handle Channels (Ensure H, W, 3)
    if img_full.ndim == 2:
        img_full = img_full[..., None].repeat(3, axis=2)
    elif img_full.shape[2] > 3:
        img_full = img_full[:, :, :3]

    print(f"[Info] Image Shape: {img_full.shape}")

    # Prepare visualization image (BGR for OpenCV)
    if img_full.max() <= 1.0:
        vis_img = (img_full * 255).astype(np.uint8)
    else:
        vis_img = img_full.astype(np.uint8)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # C. Sliding Window Inference
    H, W, _ = img_full.shape
    all_boxes = []
    all_scores = []

    print("[Info] Starting patch-based detection...")
    
    # Iterate over the image
    for y in range(0, H, STRIDE):
        for x in range(0, W, STRIDE):
            # Define patch coordinates
            y_end = min(y + PATCH_SIZE, H)
            x_end = min(x + PATCH_SIZE, W)
            
            # Crop patch
            patch = img_full[y:y_end, x:x_end]
            
            # Skip if patch is too small (edge cases)
            if patch.shape[0] < 50 or patch.shape[1] < 50:
                continue

            # Preprocess patch (Normalize & Tensor)
            patch_tensor = patch.astype(np.float32) / 255.0
            patch_tensor = torch.from_numpy(patch_tensor).permute(2, 0, 1).unsqueeze(0).to(device)

            # Inference
            with torch.no_grad():
                prediction = model(patch_tensor)[0]

            # Parse results
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()

            # Filter by confidence threshold and adjust coordinates
            for box, score in zip(boxes, scores):
                if score >= SCORE_THRESHOLD:
                    # Adjust local patch coordinates to global image coordinates
                    global_box = [
                        box[0] + x, # xmin
                        box[1] + y, # ymin
                        box[2] + x, # xmax
                        box[3] + y  # ymax
                    ]
                    all_boxes.append(global_box)
                    all_scores.append(score)

    print(f"[Info] Raw detections count: {len(all_boxes)}")

    # D. Apply Global NMS
    # Merge overlapping boxes from different patches
    final_boxes, final_scores = apply_nms_globally(all_boxes, all_scores, iou_thresh=IOU_THRESHOLD)
    
    print(f"[Result] Final detected trees: {len(final_boxes)}")

    # E. Save Results (Visualization & CSV)
    csv_data = []

    for i, (box, score) in enumerate(zip(final_boxes, final_scores)):
        x1, y1, x2, y2 = map(int, box)
        
        # 1. Draw on image
        # Green box, thickness 2
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Label text
        label = f"{score:.2f}"
        cv2.putText(vis_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 2. Append to CSV data (Pixel Coordinates)
        # You can add reverse coordinate transformation here if needed
        csv_data.append({
            "id": i + 1,
            "xmin": x1,
            "ymin": y1,
            "xmax": x2,
            "ymax": y2,
            "score": score
        })

    # Save Image
    cv2.imwrite(OUTPUT_IMAGE_PATH, vis_img)
    print(f"[Info] Visualization saved to: {OUTPUT_IMAGE_PATH}")

    # Save CSV
    if len(csv_data) > 0:
        df = pd.DataFrame(csv_data)
        df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"[Info] Coordinates saved to: {OUTPUT_CSV_PATH}")
    else:
        print("[Warning] No trees detected. No CSV file created.")

if __name__ == "__main__":
    detect_trees()