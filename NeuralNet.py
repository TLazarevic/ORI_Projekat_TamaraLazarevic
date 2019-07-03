import os
from time import time
from typing import List

import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

# imgs = []
# labels1: List[str]=[]
# path = "C:/Users/DELL/Documents/Tamara faks/ORI/trening_skup"
# valid_images = [".jpg",".gif",".png",".tga"]
# for f in os.listdir(path):
#     ext = os.path.splitext(f)[1]
#     if ext.lower() not in valid_images:
#         continue
#     img = cv2.imread(os.path.join(path,f))  # ucitavanje slike sa
#     imgs.append(img)
#    # imgs.append(Image.open(os.path.join(path,f)))
#     f=f.split("_")[0]
#     labels1.append(int(f))
#     print(f)

#--------------BALANSIRANJE DATA SETA------------------
data_dir = 'C:/Users/DELL/Documents/Tamara faks/ORI/trening_skup'

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight




def load_split_train_test(datadir, valid_size = .2):            #organizacija trening/validacionog skupa
    train_transforms = transforms.Compose([transforms.Resize([30,30]),
                                          transforms.Grayscale(),
                                       transforms.ToTensor(),
                                           transforms.Normalize([0.5], [0.5]),

                                           lambda x: x >= 0,
                                           lambda x: x.float(),
                                           transforms.Normalize(mean=[ 0.5],
                                                                std=[ 0.225])


                                       ])
    test_transforms = transforms.Compose([transforms.Resize([30,30]),
                                          transforms.Grayscale(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5], [0.5]),

                                          lambda x: x > 0,
                                          lambda x: x.float(),
                                          transforms.Normalize(mean=[ 0.5],
                                                               std=[ 0.225])

                                      ])
    train_data = datasets.ImageFolder(datadir,
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    weights = make_weights_for_balanced_classes(train_data.imgs, len(train_data.classes))
    weights = torch.DoubleTensor(weights)

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = WeightedRandomSampler(weights, len(weights))
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=8)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=8)
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, .2)


device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")



input_size = 900 #30x30  za svaku sliku
hidden_sizes = [300, 100]
output_size = 13 #6 figura svake boje+prazno polje

criterion = nn.NLLLoss()

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))            #multiklasifikacioni problem-logsoftmax -> zbir verovatnoca je 1,visa vrednost=veca vrvtnoca
print(model)

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
epochs = 30
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        # Flatten images into a 900 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)

        # This is where the model learns by backpropagating
        loss.backward()

        # And optimizes its weights here
        optimizer.step()

        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
print("\nTraining Time (in minutes) =", (time() - time0) / 60)

torch.save(model, './my_chess_model.pt')

correct_count, all_count = 0, 0
for images, labels in testloader:
    for i in range(len(labels)):
        img = images[i].view(1, 900)
        with torch.no_grad():
            logps = model(img)

        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        print(true_label,pred_label)
        if (true_label == pred_label):
            correct_count += 1
        all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count / all_count))
