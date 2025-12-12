import torch
from torchvision import models
import torch.nn.functional as F
import tifffile as tiff
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- Configuration ---
IMG_PATH = "Dataset/Orange_trees.tif"
XML_PATH = "Dataset/Orange_trees.xml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def create_pascal_voc_xml(filename, shape, boxes, output_path):
    """
    Creates an XML file in PASCAL VOC format from detected boxes.
    """
    height, width, depth = shape
    
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'Dataset'
    ET.SubElement(annotation, 'filename').text = filename
    
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    for box in boxes:
        # box: [x, y, w, h] -> need [xmin, ymin, xmax, ymax]
        x, y, w, h = box
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = 'orange_tree'
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(x)
        ET.SubElement(bndbox, 'ymin').text = str(y)
        ET.SubElement(bndbox, 'xmax').text = str(x + w)
        ET.SubElement(bndbox, 'ymax').text = str(y + h)

    # Save pretty printed XML
    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    with open(output_path, "w") as f:
        f.write(xml_str)
    print(f"✅ XML file successfully created at: {output_path}")

def nms(boxes, scores, iou_threshold=0.3):
    # Standard Non-Maximum Suppression
    if len(boxes) == 0: return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def generate_auto_labels():
    # 1. Load Image
    if not os.path.exists(IMG_PATH):
        print("Error: Image not found!")
        return

    img = tiff.imread(IMG_PATH)
    if img.ndim == 2: img = img[..., None].repeat(3, axis=2)
    elif img.shape[2] > 3: img = img[:, :, :3]
    
    # Keep original shape for XML
    orig_shape = img.shape
    
    # Preprocess
    img_norm = img.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    H, W = img_t.shape[2], img_t.shape[3]

    # 2. Setup Feature Extractor (ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    model.eval()

    # 3. Extract Reference Feature (Using the center of the image as a 'guess')
    # Or ideally, crop a known tree manually. Here we assume center crop is a tree.
    center_y, center_x = H // 2, W // 2
    patch_size = 224
    
    # Safety check for image size
    if H < patch_size or W < patch_size:
        print("Image is too small for this patch size.")
        return

    ref_patch = img_t[:, :, center_y:center_y+patch_size, center_x:center_x+patch_size]
    
    with torch.no_grad():
        reference_feature = feature_extractor(ref_patch).view(1, -1)

    print("Extracting features and generating labels... (This may take a minute)")
    
    # 4. Sliding Window Detection
    stride = 112
    detections = []
    
    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch = img_t[:, :, y:y+patch_size, x:x+patch_size]
                feat = feature_extractor(patch).view(1, -1)
                sim = F.cosine_similarity(feat, reference_feature).item()
                
                # Threshold: Adjust this if too many/few boxes are created
                if sim > 0.85: 
                    detections.append((x, y, patch_size, patch_size, sim))

    # 5. Apply NMS
    boxes = [(x, y, w, h) for (x, y, w, h, s) in detections]
    scores = [s for (x, y, w, h, s) in detections]
    
    keep_idx = nms(boxes, scores, iou_threshold=0.3)
    final_boxes = [boxes[i] for i in keep_idx]
    
    print(f"Found {len(final_boxes)} potential trees.")

    # 6. Save to XML
    if len(final_boxes) > 0:
        create_pascal_voc_xml("Orange_trees.tif", orig_shape, final_boxes, XML_PATH)
    else:
        print("No trees found. Try lowering the threshold (0.85) in the code.")

if __name__ == "__main__":
    generate_auto_labels()