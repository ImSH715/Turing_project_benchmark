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
PATCH_SIZE = 1000   # 이미지를 자를 크기 (1000x1000)
STRIDE = 800        # 겹치면서 이동할 간격 (겹쳐야 경계선에 있는 나무도 찾음)
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print(f"Using device: {DEVICE}")

def get_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def apply_nms(orig_prediction, iou_thresh=0.3):
    """겹치는 박스 제거 (Non-Maximum Suppression)"""
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)
    final_prediction = {
        'boxes': orig_prediction['boxes'][keep],
        'scores': orig_prediction['scores'][keep],
        'labels': orig_prediction['labels'][keep]
    }
    return final_prediction

def detect_large_image():
    # 1. 모델 로드
    num_classes = 2
    model = get_faster_rcnn_model(num_classes)
    
    if os.path.exists(MODEL_PATH):
        # map_location을 사용하여 GPU/CPU 불일치 방지
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {MODEL_PATH}")
    else:
        print("Model weights not found!")
        return

    model.to(DEVICE)
    model.eval()

    # 2. 이미지 로드
    img_full = tiff.imread(IMAGE_PATH)
    print(f"Original image shape: {img_full.shape}")

    # 채널 정리 (H, W, C)
    if img_full.ndim == 2:
        img_full = img_full[..., None].repeat(3, axis=2)
    elif img_full.shape[2] > 3:
        img_full = img_full[:, :, :3]
    
    # 시각화용 이미지 (Uint8)
    if img_full.max() <= 1.0:
        vis_img = (img_full * 255).astype(np.uint8)
    else:
        vis_img = img_full.astype(np.uint8)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # 3. 슬라이딩 윈도우 (Patch Detection)
    H, W, _ = img_full.shape
    all_boxes = []
    all_scores = []
    
    print("Starting patch-based detection...")
    
    # 세로(y)와 가로(x)로 이동하며 자르기
    for y in range(0, H, STRIDE):
        for x in range(0, W, STRIDE):
            # 패치 영역 계산 (이미지 끝부분 처리)
            y_end = min(y + PATCH_SIZE, H)
            x_end = min(x + PATCH_SIZE, W)
            
            # 실제 패치 자르기
            patch = img_full[y:y_end, x:x_end]
            
            # 패치가 너무 작으면 스킵
            if patch.shape[0] < 50 or patch.shape[1] < 50:
                continue

            # 전처리 (0~1 float, Tensor변환)
            patch_tensor = patch.astype(np.float32) / 255.0
            patch_tensor = torch.from_numpy(patch_tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

            # 추론
            with torch.no_grad():
                prediction = model(patch_tensor)[0]

            # 결과 좌표 보정 (패치 내부 좌표 -> 전체 이미지 좌표)
            boxes = prediction['boxes'].cpu().numpy()
            scores = prediction['scores'].cpu().numpy()
            
            for box, score in zip(boxes, scores):
                if score >= SCORE_THRESHOLD:
                    # 박스 좌표에 현재 패치의 시작 위치(x, y)를 더해줌
                    real_box = [
                        box[0] + x, 
                        box[1] + y, 
                        box[2] + x, 
                        box[3] + y
                    ]
                    all_boxes.append(real_box)
                    all_scores.append(score)

    # 4. 전체 결과에 대해 NMS 적용 (중복 박스 제거)
    if len(all_boxes) > 0:
        all_boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
        all_scores_t = torch.tensor(all_scores, dtype=torch.float32)
        all_labels_t = torch.ones(len(all_scores), dtype=torch.int64) # 라벨은 모두 1(Tree)

        # 딕셔너리로 만들어서 NMS 함수에 전달
        pred_dict = {'boxes': all_boxes_t, 'scores': all_scores_t, 'labels': all_labels_t}
        final_result = apply_nms(pred_dict, iou_thresh=0.2)
        
        final_boxes = final_result['boxes'].numpy()
        final_scores = final_result['scores'].numpy()
        
        print(f"Detected {len(final_boxes)} trees after merging patches.")

        # 5. 그리기
        for box, score in zip(final_boxes, final_scores):
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3) # 빨간색 박스
    else:
        print("No trees detected.")

    # 6. 저장
    cv2.imwrite(OUTPUT_PATH, vis_img)
    print(f"Result saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    detect_large_image()