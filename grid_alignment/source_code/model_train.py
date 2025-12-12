import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import random
import os
class PatchDataset(Dataset):
    """Dataset that samples random patches from a large image."""
    def __init__(self, img_path, patch_size=128, num_patches=20000):
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.image = Image.open(img_path).convert("RGB")
        self.W, self.H = self.image.size

        # SimCLR augmentation pipeline
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(patch_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        # Sample random crop coordinates
        x = random.randint(0, self.W - self.patch_size)
        y = random.randint(0, self.H - self.patch_size)
        img_patch = self.image.crop((x, y, x+self.patch_size, y+self.patch_size))

        # Two augmented views (SimCLR)
        return self.transform(img_patch), self.transform(img_patch)
class SimCLR(nn.Module):
    """SimCLR model using ResNet as encoder and MLP projection head."""
    def __init__(self, base_model="resnet50", out_dim=128):
        super().__init__()
        
        resnet = models.__dict__[base_model](pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # remove FC

        # Dimensions: ResNet50 outputs 2048-D
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projection_head(h)
        return F.normalize(z, dim=1)
def nt_xent_loss(z1, z2, temperature=0.5):
    """Normalized Temperature-scaled Cross Entropy Loss."""
    batch = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2B x D

    # Cosine similarity
    sim = torch.mm(z, z.t()) / temperature

    # Mask: remove similarity to itself
    mask = (~torch.eye(2 * batch, 2 * batch, dtype=torch.bool)).to(z.device)
    sim = sim.masked_select(mask).view(2 * batch, -1)

    positives = torch.sum(z1 * z2, dim=-1) / temperature
    positives = torch.cat([positives, positives], dim=0)

    labels = torch.zeros(2 * batch).long().to(z.device)

    loss = F.cross_entropy(sim, labels)
    return loss
def train_simclr(img_path, epochs=10, batch_size=64, lr=3e-4, save_path="simclr_model.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = PatchDataset(img_path, patch_size=128, num_patches=20000)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    model = SimCLR().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)

            z1 = model(x1)
            z2 = model(x2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss / len(loader):.4f}")

    # Save only encoder (feature extractor)
    torch.save(model.encoder.state_dict(), save_path)
    print(f"\nSaved feature encoder to: {save_path}")
train_simclr("../Dataset/Orange_trees.tif", epochs=20, batch_size=128)
