import os
import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

IMG_DIR = "Dataset/tiles/images"
MASK_DIR = "Dataset/tiles/masks"
MODEL_OUT = "unet_tree_crown.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TileDataset(Dataset):
    def __init__(self):
        self.images = sorted(os.listdir(IMG_DIR))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(IMG_DIR, self.images[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(
            os.path.join(MASK_DIR, self.images[idx].replace("img", "mask")),
            cv2.IMREAD_GRAYSCALE
        )
        mask = (mask > 0).astype(np.float32)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def C(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU()
            )
        self.d1 = C(3, 64)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = C(64, 128)
        self.p2 = nn.MaxPool2d(2)
        self.b = C(128, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3 = C(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d4 = C(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.p1(c1))
        b = self.b(self.p2(c2))
        d3 = self.d3(torch.cat([self.u2(b), c2], 1))
        d4 = self.d4(torch.cat([self.u1(d3), c1], 1))
        return torch.sigmoid(self.out(d4))

dataset = TileDataset()
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = UNet().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.BCELoss()

epochs = 100

for epoch in range(epochs):
    model.train()
    total = 0
    for img, mask in loader:
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        pred = model(img)
        loss = loss_fn(pred, mask)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {epoch+1} Loss {total:.4f}")

torch.save(model.state_dict(), MODEL_OUT)
print("Model saved:", MODEL_OUT)
