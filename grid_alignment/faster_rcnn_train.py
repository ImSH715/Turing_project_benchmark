import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff
import numpy as np
import os
import cv2
import pandas as pd

# --- Configuration ---
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# --- Constants for Coordinate Conversion ---
# [CRITICAL] The coordinates in your CSV look like Map Coordinates (UTM), not Pixel Coordinates.
# You MUST set these values based on your TIF file metadata (e.g., check in QGIS or using gdalinfo).
# If your CSV already contains pixel coordinates (0 to ImageWidth), set ORIGIN to 0 and PIXEL_SIZE to 1.
ORIGIN_X = 621370.0      # Example: Top-Left X coordinate of the TIF (Adjust this!)
ORIGIN_Y = 7741205.0     # Example: Top-Left Y coordinate of the TIF (Adjust this!)
PIXEL_SIZE = 0.1         # Example: Resolution (meters per pixel) (Adjust this!)
BOX_SIZE = 40            # Size of the square box to draw around the point (in pixels)

# --- 1. CSV Parsing Function ---
def load_boxes_from_csv(csv_path):
    """
    Reads a CSV file with 'lat' and 'long' columns.
    Converts points to bounding boxes.
    Assumes: lat = x, long = y (as per user instruction).
    """
    if not os.path.exists(csv_path):
        print(f"[Error] CSV file not found: {csv_path}")
        return []
    
    # Read CSV using pandas
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return []

    boxes = []
    
    for index, row in df.iterrows():
        # User specified: lat is x, long is y
        map_x = float(row['lat'])
        map_y = float(row['long'])
        
        # --- Coordinate Transformation (Map -> Pixel) ---
        # Calculate pixel coordinates from map coordinates
        # Assumes TIF origin is Top-Left.
        # Pixel X = (Current Map X - Origin X) / Pixel Size
        # Pixel Y = (Origin Y - Current Map Y) / Pixel Size (Y usually inverted)
        pixel_x = (map_x - ORIGIN_X) / PIXEL_SIZE
        pixel_y = (ORIGIN_Y - map_y) / PIXEL_SIZE 
        
        # Create a bounding box around the center point
        half_size = BOX_SIZE / 2
        xmin = pixel_x - half_size
        ymin = pixel_y - half_size
        xmax = pixel_x + half_size
        ymax = pixel_y + half_size
        
        boxes.append([xmin, ymin, xmax, ymax])
        
    return boxes

# --- 2. Dataset Class ---
class OrangeTreeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_path = os.path.join(root_dir, "Orange_trees.tif")
        # Changing expected file from XML to CSV
        self.csv_path = os.path.join(root_dir, "ground_truth.csv")
        
        # Check if file exists
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"[Critical] Required file missing: {self.csv_path} (Please prepare the CSV file first!)")

    def __len__(self):
        return 1 

    def __getitem__(self, idx):
        # 1. Load Image
        img_np = tiff.imread(self.img_path)

        # Handle Channels (Ensure H, W, 3)
        if img_np.ndim == 2:
            img_np = img_np[..., None].repeat(3, axis=2)
        elif img_np.shape[2] > 3:
            img_np = img_np[:, :, :3]
            
        # Normalize and Convert to Tensor
        img_np = img_np.astype(np.float32) / 255.0
        image = torch.from_numpy(img_np).permute(2, 0, 1) # (C, H, W)

        # 2. Load Labels from CSV
        boxes = load_boxes_from_csv(self.csv_path)
        
        # Handle empty boxes to prevent errors
        if len(boxes) == 0:
            print("[Warning] No labels (boxes) found in CSV.")
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64) # Class 1 (Tree)
            
            # Calculate area
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target

# --- 3. Model Definition ---
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# --- 4. Data Verification Function ---
def check_data_before_training(dataset):
    print("Checking data labeling status...")
    img_t, target = dataset[0]
    
    # Convert Tensor to Numpy Image for visualization
    img_np = img_t.permute(1, 2, 0).numpy()
    img_vis = (img_np * 255).astype(np.uint8).copy()
    img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    
    boxes = target["boxes"].numpy()
    print(f"[Info] Found {len(boxes)} boxes.")
    
    if len(boxes) == 0:
        print("[Error] No boxes found! Check your CSV file.")
        return False

    # Draw boxes
    h, w, _ = img_vis.shape
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Check if coordinates are within image bounds
        # This is crucial for checking if coordinate conversion is correct
        if x1 < 0 or x1 > w or y1 < 0 or y1 > h:
            pass # You might want to print a warning here if boxes are way off
        
        cv2.rectangle(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 5) # Blue box
        
    cv2.imwrite("Check_Label_CSV.png", img_vis)
    print("Saved verification image: Check_Label_CSV.png (Check if blue boxes are aligned with trees!)")
    return True

# --- 5. Training Function ---
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0
    
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # 1. Prepare Dataset
        # Ensure 'Orange_trees.csv' exists in the Dataset folder
        dataset = OrangeTreeDataset(root_dir="Dataset/")
        
        # 2. Verify Data (Check if CSV is loaded correctly)
        # It is highly recommended to check 'Check_Label_CSV.png' before full training
        if not check_data_before_training(dataset):
            print("[Critical] Training aborted: Data labeling issue.")
            exit()

        data_loader = DataLoader(
            dataset, batch_size=1, shuffle=True, 
            collate_fn=lambda x: tuple(zip(*x))
        )

        # 3. Setup Model
        num_classes = 2 # 1(Tree) + 1(Background)
        model = get_model(num_classes)
        model.to(device)

        # 4. Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

        # 5. Start Training
        num_epochs = 150
        print("Start Training...")
        for epoch in range(num_epochs):
            train_one_epoch(model, optimizer, data_loader, device, epoch)
        
        # 6. Save Model
        torch.save(model.state_dict(), "faster_rcnn_orange.pth")
        print("Model saved successfully: faster_rcnn_orange.pth")
        
    except Exception as e:
        print(f"\n[Error] Exception occurred: {e}")
        print("Please check if 'Orange_trees.csv' exists in the Dataset folder and format is correct.")