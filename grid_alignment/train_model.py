import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import tifffile as tiff
import numpy as np
import os
import cv2
import pandas as pd

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
        self.csv_path = os.path.join(root_dir, "random_point.csv")
        
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