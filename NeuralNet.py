import os
from time import time
from typing import List

import cv2
import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import scipy
from scipy import misc

class MyHOG(object):
    def __call__(self, input):
        target=hogF(input)
        return target

def hogF(im):
    im = np.asarray(im, dtype="int32")
    im = np.array(im * 255, dtype=np.uint8)

    nbins = 9  # broj binova
    cell_size = (4, 4)  # broj piksela po celiji
    block_size = (2, 2)  # broj celija po bloku

    h=cv2.HOGDescriptor(_winSize=(32 // cell_size[1] * cell_size[1],
                                  32 // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)
    return h.compute(im)
#--------------BALANSIRANJE DATA SETA------------------
data_dir = 'C:/Users/DELL/Documents/Tamara faks/ORI/trening_skup/'
data_dir2 = 'C:/Users/DELL/Documents/Tamara faks/ORI/test_skup/'

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




def load_split_train_test(datadir,datadir2, valid_size = .2):            #organizacija trening/validacionog skupa
    train_transforms = transforms.Compose([transforms.Resize([32,32]),
                                          transforms.Grayscale(),
                                           MyHOG(),
                                           transforms.ToTensor(),
                                       #transforms.ToTensor(),
                                           # transforms.Normalize([0.5], [0.5]),

                                           #  lambda x: x >= 0,
                                           #  lambda x: x.float(),
                                           # transforms.Normalize(mean=[ 0.5],
                                           #                      std=[ 0.5])




                                       ])
    test_transforms = transforms.Compose([transforms.Resize([32,32]),
                                         transforms.Grayscale(),
                                          MyHOG(),
                                          transforms.ToTensor(),
                                          #transforms.ToTensor(),
                                          # transforms.Normalize([0.5], [0.5]),


                                          #  lambda x: x >= 0,
                                          #  lambda x: x.float(),
                                          # transforms.Normalize(mean=[ 0.5],
                                          #                      std=[ 0.5])

                                      ])
    t2=transforms.Compose([ transforms.ToTensor(),

    ])

    train_data = datasets.ImageFolder(datadir,
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir2,
                    transform=test_transforms)

    tr2=datasets.VisionDataset(train_data,transform=t2)
    # for images, labels in train_data:
    #
    #       #  images[i] = transforms.Resize([30, 30])
    #         _, images = cv2.threshold(images, 127, 255, cv2.THRESH_OTSU)
    #        # images[i] = transforms.ToTensor()
    #
    # for images, labels in test_data:
    #
    #        # images[i] = transforms.Resize([30, 30])
    #         _, images = cv2.threshold(images, 127, 255, cv2.THRESH_OTSU)
    #        # images[i] = transforms.ToTensor()
    #
    # train_data=train_transforms(train_data)
    # test_data=test_transforms(test_data)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    weights = make_weights_for_balanced_classes(train_data.imgs, len(train_data.classes))
    weights = torch.DoubleTensor(weights)

    weights2=make_weights_for_balanced_classes(test_data.imgs, len(test_data.classes))
    weights2=torch.DoubleTensor(weights2)

    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = WeightedRandomSampler(weights, len(weights))
    test_sampler = WeightedRandomSampler(weights2,len(weights2))
    trainloader = torch.utils.data.DataLoader(train_data,sampler=train_sampler, batch_size=8)
    print(trainloader)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=8) #brze jer ne koristi grad
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir,data_dir2, .2)



# device = torch.device("cuda" if torch.cuda.is_available()
#                                   else "cpu")



input_size = 1764 #30x30  za svaku sliku
hidden_sizes = [400, 200]
output_size = 13 #6 figura svake boje+prazno polje

criterion = nn.NLLLoss()

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                     # nn.Dropout(p=0.5),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                    # nn.Dropout(p=0.5),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))            #multiklasifikacioni problem-logsoftmax -> zbir verovatnoca je 1,visa vrednost=veca vrvtnoca
print(model)
print(trainloader.__len__())
print(testloader.__len__())

optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9,weight_decay=0.0001)
time0 = time()
epochs = 35

trlo=[]
testlo=[]

for e in range(epochs):
    running_loss = 0
    model.train()
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

        optimizer.zero_grad()

    else:
        model.eval()
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
        trlo.append(running_loss / len(trainloader))

        with torch.no_grad():
                valid_loss = sum(criterion(model(xb.view(xb.shape[0], -1)), yb) for xb, yb in testloader)
                testlo.append(valid_loss / len(testloader))
        print(e, valid_loss / len(testloader))

print("\nTraining Time (in minutes) =", (time() - time0) / 60)

torch.save(model, './my_chess_model.pt')

plt.plot(trlo, label='Training loss')
plt.plot(testlo, label='Validation loss')
plt.legend(frameon=False)
plt.show()

correct_count, all_count = 0, 0
for images, labels in testloader:
    for i in range(len(labels)):

        img = images[i].view(1, 1764)

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
