# ------------------- train_faster_rcnn.py ONLY (improved) -------------------

"""
Improved Faster R-CNN training script with:
- better pseudo-label segmentation (vegetation index)
- more robust bbox filtering
- SGD optimizer with tuned LR and scheduler
- IoU-based accuracy (recall) with greedy matching

Main dataset (big TIFF): ../Dataset/Orange_trees.tif
Training tiles:        ../Dataset/data/*.tif

Usage: just run `python train_faster_rcnn.py`
"""

import os
import glob
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader

# dataset paths
MAIN_TIF = "../Dataset/Orange_trees.tif"
DEFAULT_IMAGE_DIR = "../Dataset/data"

# -------------------------------------------------------------
# segmentation (improved)
# -------------------------------------------------------------
def segment_canopies(img):
    # img: RGB ndarray
    R = img[:,:,0].astype(np.int16)
    G = img[:,:,1].astype(np.int16)
    B = img[:,:,2].astype(np.int16)

    # normalized green-ish index (simple): 2*G - R - B
    ng = 2*G - R - B
    mask = (ng > 20) & (G > 40)  # empirical thresholds
    mask = mask.astype('uint8') * 255

    # slightly smaller morphology to preserve canopy shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask

# -------------------------------------------------------------
# dataset
# -------------------------------------------------------------
class PseudoBBoxDataset(Dataset):
    def __init__(self, image_files, min_area=1000):
        self.image_files = image_files
        self.min_area = min_area

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise RuntimeError(f'Cannot read image: {path}')

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = segment_canopies(img)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            # use boundingRect but filter by area and aspect ratio if needed
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area >= self.min_area:
                # optional: ignore extremely elongated boxes
                ar = float(w)/float(h+1e-6)
                if ar < 0.2 or ar > 5.0:
                    continue
                boxes.append([x, y, x+w, y+h])

        # convert to tensors (if no boxes, create empty tensors)
        if len(boxes) == 0:
            boxes_t = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((boxes_t.shape[0],), dtype=torch.int64)  # class 1 = tree

        image_tensor = torchvision.transforms.functional.to_tensor(Image.fromarray(img))

        target = {
            'boxes': boxes_t,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'iscrowd': torch.zeros((boxes_t.shape[0],), dtype=torch.int64)
        }

        return image_tensor, target

# -------------------------------------------------------------
# model
# -------------------------------------------------------------
def get_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # background + tree
    return model


def collate_fn(batch):
    return tuple(zip(*batch))

# -------------------------------------------------------------
# IoU and matching (greedy)
# -------------------------------------------------------------

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    areaA = max(0.0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0.0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_predictions_to_gts(pred_boxes, gt_boxes, iou_threshold=0.3):
    # pred_boxes, gt_boxes: numpy arrays Nx4, Mx4
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0, 0, 0  # tp, fp, fn

    used_pred = set()
    tp = 0

    for gi, gb in enumerate(gt_boxes):
        best_iou = 0.0
        best_p = -1
        for pi, pb in enumerate(pred_boxes):
            if pi in used_pred:
                continue
            cur_iou = iou(pb, gb)
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_p = pi
        if best_iou >= iou_threshold:
            tp += 1
            used_pred.add(best_p)

    fp = len(pred_boxes) - len(used_pred)
    fn = len(gt_boxes) - tp
    return tp, fp, fn

# -------------------------------------------------------------
# training
# -------------------------------------------------------------

def train(images_dir, out_path, epochs=100, batch_size=2):
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.tif')))
    assert len(image_files) > 0, "No .tif images found in folder"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    dataset = PseudoBBoxDataset(image_files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             num_workers=2, collate_fn=collate_fn)

    model = get_model().to(device)
    params = [p for p in model.parameters() if p.requires_grad]

    # SGD optimizer (tuned)
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tp = 0
        epoch_fp = 0
        epoch_fn = 0

        for images, targets in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            images = [img.to(device) for img in images]
            targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]

            losses = model(images, targets_gpu)
            loss = sum(v for v in losses.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # evaluate predictions for basic detection metrics (per-batch)
            with torch.no_grad():
                model.eval()
                outputs = model(images)
                model.train()

                for out, tgt in zip(outputs, targets):
                    pred_boxes = out.get('boxes', torch.zeros((0,4))).cpu().numpy()
                    gt_boxes = tgt.get('boxes', torch.zeros((0,4))).cpu().numpy()

                    tp, fp, fn = match_predictions_to_gts(pred_boxes, gt_boxes, iou_threshold=0.3)
                    epoch_tp += tp
                    epoch_fp += fp
                    epoch_fn += fn

        scheduler.step()

        precision = epoch_tp / (epoch_tp + epoch_fp) if (epoch_tp + epoch_fp) > 0 else 0.0
        recall = epoch_tp / (epoch_tp + epoch_fn) if (epoch_tp + epoch_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}  Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    # final save
    torch.save(model.state_dict(), out_path)
    print(f"Training complete. Model saved → {out_path}")

# -------------------------------------------------------------
# Auto-run
# -------------------------------------------------------------
if __name__ == '__main__':
    IMAGE_DIR = DEFAULT_IMAGE_DIR
    OUT_PATH = 'model_final.pth'
    EPOCHS = 100
    BATCH = 2

    print(f"Using predefined settings: images={IMAGE_DIR} out={OUT_PATH} epochs={EPOCHS} batch={BATCH}")
    train(IMAGE_DIR, OUT_PATH, EPOCHS, BATCH)
