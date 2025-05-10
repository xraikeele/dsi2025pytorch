import time
import torch
import torchvision
import timm

import os
from pathlib import Path

from torchvision import transforms
from tqdm import tqdm

import torch.optim as optim
import torch.nn as nn

BATCH_SIZE = 64
EPOCHS = 5
WORKERS = 2
IMG_DIMS = (336, 336)
CLASSES = 10

MODEL_NAME = 'resnet50d'

datadir = Path('/home/dsi2025/cifar10/')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_DIMS),
])

data = torchvision.datasets.CIFAR10(datadir,
                                    train=True,
                                    download=True,
                                    transform=transform)
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=WORKERS)

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=CLASSES)

print("Model Created")

device = torch.device('cuda:0')
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

start = time.perf_counter()
for epoch in range(EPOCHS):
    epoch_start_time = time.perf_counter()

    model.train()
    for batch in tqdm(data_loader, total=len(data_loader)):
        features, labels = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        preds = model(features)
        loss = loss_fn(preds, labels)

        loss.backward()
        optimizer.step()

    epoch_end_time = time.perf_counter()
    print(f"Epoch {epoch+1} Time", epoch_end_time - epoch_start_time)
end = time.perf_counter()
print("Training Took", end - start)