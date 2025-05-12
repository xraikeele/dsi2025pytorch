import os
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

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

dist.init_process_group("nccl")

local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])

BATCH_SIZE = 256 // int(os.environ["WORLD_SIZE"])
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
sampler = DistributedSampler(data)
data_loader = torch.utils.data.DataLoader(data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False,
                                          sampler=sampler,
                                          num_workers=WORKERS)

if global_rank == 0:
    print("Master address is:", os.environ["MASTER_ADDR"])
    print(f"Local Rank: {local_rank}")

torch.cuda.set_device(local_rank)
torch.cuda.empty_cache()

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=CLASSES)

model = model.to('cuda:' + str(local_rank))
model = DDP(model, device_ids=[local_rank])

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

start = time.perf_counter()
for epoch in range(EPOCHS):
    epoch_start_time = time.perf_counter()

    model.train()
    for batch in tqdm(data_loader, total=len(data_loader)):
        features, labels = batch[0].to(local_rank), batch[1].to(local_rank)

        optimizer.zero_grad()

        preds = model(features)
        loss = loss_fn(preds, labels)

        loss.backward()
        optimizer.step()

    epoch_end_time = time.perf_counter()
    if global_rank == 0:
        print(f"Epoch {epoch+1} Time", epoch_end_time - epoch_start_time)
end = time.perf_counter()

dist.destroy_process_group()

if global_rank == 0:
    print("Training Took", end - start)
