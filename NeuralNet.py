import os
from typing import List

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

imgs = []
labels: List[str]=[]
path = "C:/Users/DELL/Documents/Tamara faks/ORI/trening_skup"
valid_images = [".jpg",".gif",".png",".tga"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    img = cv2.imread(os.path.join(path,f))  # ucitavanje slike sa
    imgs.append(img)
   # imgs.append(Image.open(os.path.join(path,f)))
    f=f.split("_")[0]
    labels.append(f)
    print(f)

n_in, n_h, n_out, batch_size = 400, 5, 8, 10  # layer size and batch size
x = torch.tensor(imgs)
y = torch.tensor(labels)
model = nn.Sequential(nn.Linear(n_in, n_h),
                      nn.ReLU(),
                      nn.Linear(n_h, n_out),
                      nn.Sigmoid())
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(50):
    # Forward Propagation
    y_pred = model(x)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch, ' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()

    # perform a backward pass (backpropagation)
    loss.backward()

    # Update the parameters
    optimizer.step()
