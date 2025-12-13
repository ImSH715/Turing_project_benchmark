import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import tifffile as tiff
import numpy as np
import cv2
import os

# --- Configuration ---
MODEL_PATH = "faster_rcnn_orange.pth"
IMAGE_PATH = "Dataset/Orange_trees.tif"
OUTPUT_PATH = "Detected_Output_Patch.png"
SCORE_THRESHOLD = 0.5
PATCH_SIZE = 1000   
STRIDE = 800     
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f"Using device: {DEVICE}")

def get_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def apply_nms(orig_prediction, iou_thresh=0.3):
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = {
        'boxes': orig_prediction['boxes'][keep],
        'scores': orig_prediction['scores'][keep],
        'labels': orig_prediction['labels'][keep]
    }
    return final_prediction

def detect_large_image():
    num_classes = 2
    model = get_faster_rcnn_model(num_classes)
    
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {MODEL_PATH}")
    else:
        print("Model weights not found!")
        return

    model.to(DEVICE)
    model.eval()

    img_full = tiff.imread(IMAGE_PATH)
    print(f"Original image shape: {img_full.shape}")

    if img_full.ndim == 2:
        img_full = img_full[..., None].repeat(3, axis=2)
    elif img_full.shape[2] > 3:
        img_full = img_full[:, :, :3]
    
    if img_full.max() <= 1.0:
        vis_img = (img_full * 255).astype(np.uint8)
    else:
        vis_img = img_full.astype(np.uint8)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    H, W, _ = img_full.shape
    all_boxes = []
    all_scores = []
    
    print("Starting patch-based detection...")
    
    for y in range(0, H, STRIDE):
        for x in range(0, W, STRIDE):
            y_end = min(y + PATCH_SIZE, H)
            x_end = min(x + PATCH_SIZE, W)
            
            patch = img_full[y:y_end, x:x_end]
            
            if patch.shape[0] < 50 or patch.shape[1] < 50:
                continue

            patch_tensor = patch.astype(np.float32) / 255.0
            patch_tensor = torch.from_numpy(patch_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                prediction = model(patch_tensor)[0]

            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            
            for box, score in zip(boxes, scores):
                if score >= SCORE_THRESHOLD:
                    real_box = [
                        box[0] + x, 
                        box[1] + y, 
                        box[2] + x, 
                        box[3] + y
                    ]
                    all_boxes.append(real_box)
                    all_scores.append(score)

    if len(all_boxes) > 0:
        all_boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
        all_scores_t = torch.tensor(all_scores, dtype=torch.float32)
        all_labels_t = torch.ones(len(all_scores), dtype=torch.int64) # 라벨은 모두 1(Tree)

        pred_dict = {'boxes': all_boxes_t, 'scores': all_scores_t, 'labels': all_labels_t}
        final_result = apply_nms(pred_dict, iou_thresh=0.2)
        
        final_boxes = final_result['boxes'].numpy()
        final_scores = final_result['scores'].numpy()
        
        print(f"Detected {len(final_boxes)} trees after merging patches.")

        for box, score in zip(final_boxes, final_scores):
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3) # 빨간색 박스
    else:
        print("No trees detected.")

    cv2.imwrite(OUTPUT_PATH, vis_img)
    print(f"Result saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    detect_large_image()