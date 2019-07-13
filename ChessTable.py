
import numpy as np
import cv2  # OpenCV biblioteka
import matplotlib.pyplot as plt
import skimage
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import skimage.color

from torchvision import transforms
from tkinter import filedialog


class MyHOG(object):
    def __call__(self, input):
        target = hogF(input)
        return target


def hogF(im):
    im = np.asarray(im, dtype="int32")
    im = np.array(im*255 , dtype=np.uint8)


    nbins = 9  # broj binova
    cell_size = (3, 3)  # broj piksela po celiji
    block_size = (3, 3)  # broj celija po bloku

    h = cv2.HOGDescriptor(_winSize=(30 // cell_size[1] * cell_size[1],
                                    30 // cell_size[0] * cell_size[0]),
                          _blockSize=(block_size[1] * cell_size[1],
                                      block_size[0] * cell_size[0]),
                          _blockStride=(cell_size[1], cell_size[0]),
                          _cellSize=(cell_size[1], cell_size[0]),
                          _nbins=nbins)
    return h.compute(im)


# ------------------------------otvaranje slike--------------------

file_path = filedialog.askopenfilename()

img = cv2.imread(file_path)  # ucitavanje slike sa diska
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # konvertovanje iz BGR u RGB model boja (OpenCV ucita sliku kao BGR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --------pronalazenje linija na slici table-------------------

edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 230)

print(len(lines))

if(len(lines)>10000):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 360)
elif(len(lines)>900):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 340)
elif(len(lines)>500):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 320)
elif(len(lines)>300):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 350)
elif(len(lines>100)):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 270)



lines = np.unique(lines, axis=0)

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
# cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


# ------------------pronalazenje preseka linija---------------------

vertlines = lines[np.where(lines[:, :, 1] == np.pi / 2)]
horlines = lines[np.where(lines[:, :, 1] == 0)]


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.
    Returns closest integer pixel locations."""

    rho1 = line1[0]
    theta1 = line1[1]
    rho2 = line2[0]
    theta2 = line2[1]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


intersections = []
for line1 in vertlines:
    for line2 in horlines:
        intersections.append(intersection(line1, line2))


# ---------------brisanje duplih preseka-----------

fieldWid = int(
    img.shape[1] / 8)  # sirina sahovske table odgovara sirini (i duzini) osam polja i na skrinsotu i na isecenoj slici
fieldLen = int(img.shape[1] / 8)

print(fieldLen)


icopy = intersections

print("intersections: ", intersections.__len__())
print("intersections: ", intersections)

for cnt in range(2):
    for i in range((icopy.__len__() - 1), -1, -1):  # brisanje preseka koji su blizi nego velicina polja
        if (abs(intersections[i][0][0] - intersections[i - 1][0][0]) < fieldWid / 2 + 5) and (
                intersections[i][0][1] in range(intersections[i - 1][0][1] - 5, intersections[i - 1][0][1] + 5)):
            print("removed1"+str(intersections[i-1])+"reason "+str(intersections[i][0][0] - intersections[i - 1][0][0]))
            intersections.remove(intersections[i-1])

intersections = sorted(intersections, key=lambda coor: coor[0][0])
print("intersections: ", intersections.__len__())
print("intersections: ", intersections)
icopy = intersections

for cnt in range(2):
    for i in range((icopy.__len__() - 1), -1, -1):  # brisanje preseka koji su blizi nego velicina polja
        if (abs(intersections[i][0][1] - intersections[i - 1][0][1]) < fieldLen / 2 + 5) and (
                intersections[i][0][0] in range(intersections[i - 1][0][0] - 5, intersections[i - 1][0][0] + 5)):
            print("removed2" + str(intersections[i-1])+"reason "+str(intersections[i][0][1] - intersections[i - 1][0][1]))
            intersections.remove(intersections[i-1])


print("intersections: ", intersections.__len__())
print("intersections: ", intersections)
icopy = intersections

for cnt in range(3):
    for i in range((icopy.__len__() - 1), -1,
                   -1):  # brisanje preseka koji su predaleko (pogresno detektovani preseci ostatka skrinsota)
        if (abs(intersections[i][0][1] - intersections[i - 1][0][1]) > fieldLen + 15) and (
                intersections[i][0][0] in range(intersections[i - 1][0][0] - 5, intersections[i - 1][0][0] + 5)):
            if (intersections[i][0][1] > img.shape[0] / 2):
                print("removed3" + str(intersections[i])+"reason "+str((intersections[i][0][1] - intersections[i - 1][0][1])))
                intersections.remove(intersections[i])

            else:
                print("removed3" + str(intersections[i])+"reason "+str((intersections[i][0][1] - intersections[i - 1][0][1])))
                intersections.remove(intersections[i - 1])


print("intersections: ", intersections.__len__())
print("intersections: ", intersections)

# ------------------dodavanje nedetektovanih ivica slike----------------

# intersections.sort()

x = []
for i in intersections:
    x.append(i[0][0])
x = np.unique(x)

firstYrowdot = intersections[0][0][
                   1] - fieldLen  # adding this as Y coord to top row instead of zero to make space to make algorithm more flexible
if firstYrowdot < 0:
    while firstYrowdot < 0:
        firstYrowdot = firstYrowdot + 1

lastYrowdot = intersections[-1][0][1] + fieldLen

if lastYrowdot > img.shape[0]:
    while lastYrowdot > img.shape[0]:
        lastYrowdot = lastYrowdot - 1

if (intersections[-1][0][1] + fieldLen) in range(-4 + img.shape[0], img.shape[0] + 4):  # fali donja ivica(za isecene slike)
    for i in x:
        intersections.append([[i, lastYrowdot]])
        # intersections.append([[i, img.shape[1]-2]])

if (intersections[0][0][1] - fieldLen) in range(-4, 4):  # fali gornja ivica(za isecene slike)
    for i in x:
        # intersections.append([[i,2]])
        intersections.append([[i, firstYrowdot]])

intersections = sorted(intersections, key=lambda coor: coor[0][1])

y = []
for i in intersections:
    y.append(i[0][1])
y = np.unique(y)

firstXrowdot = intersections[0][0][
                   0] - fieldWid  # adding this as X coord to top row instead of zero to make space to make algorithm more flexible
if firstXrowdot < 0:
    while firstXrowdot < 0:
        firstXrowdot = firstXrowdot + 1

lastXrowdot = intersections[-1][0][0] + fieldWid
if lastXrowdot > img.shape[1]:
    while lastXrowdot > img.shape[1]:
        lastXrowdot = lastXrowdot - 1

if (intersections[-1][0][0] + fieldWid) in range(-4 + img.shape[1], img.shape[1] + 4):  # fali desna ivica
    for i in y:
        intersections.append([[lastXrowdot, i]])
        # intersections.append([[i, img.shape[0]-2]])

if (intersections[0][0][0] - fieldWid) in range(-4, 4):  # fali leva ivica
    for i in y:
        # intersections.append([[i,2]])
        intersections.append([[firstXrowdot, i]])

# ----------------------pronalazenje i filtriranje polja------------------------

sortx = sorted(intersections)
sorty = sorted(intersections, key=lambda coor: coor[0][1])

polja = []
velicinepolja = []

# _,img = cv2.threshold(gray, 50, 255, cv2.THRESH_OTSU)

for i in range(0, intersections.__len__() - 1):
    for j in range(0, intersections.__len__() - 1):
        if img[sorty[j][0][1]:sorty[j + 1][0][1], sortx[i][0][0]:sortx[i + 1][0][0]].size != 0:
            # print((sorty[j + 1][0][1] - sorty[j][0][1]) - (sortx[i + 1][0][0] - sortx[i][0][0]))
            if ((sorty[j + 1][0][1] - sorty[j][0][1]) - (sortx[i + 1][0][0] - sortx[i][0][0])) in range(-20,
                                                                                                        20):  # square check
                polja.append(img[sorty[j][0][1]:sorty[j + 1][0][1], sortx[i][0][0]:sortx[i + 1][0][0]])
                velicinepolja.append(polja[-1].size)

velicina = fieldLen * fieldWid

temp = polja.__len__()
print(polja.__len__())

for p in range(temp - 1, -1, -1):
    if not (polja[p].shape[0] * polja[p].shape[1] in range(velicina - 7000, velicina + 7000)):
        polja.pop(p)
    else:
        polja[p] = cv2.resize(polja[p], dsize=(30, 30), interpolation=cv2.INTER_CUBIC)
        polja[p] = cv2.cvtColor(polja[p], cv2.COLOR_BGR2GRAY)

print(polja.__len__())
# ------------------------plots---------------------------------
# br=0
# for p in polja:
#     #p = cv2.cvtColor(p, cv2.COLOR_GRAY2RGB)
#     # _, p = cv2.threshold(p, 127, 255, cv2.THRESH_OTSU)
#     # p = cv2.adaptiveThreshold(p, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20)
#     # p = p / 255.0
#     plt.figure()
#     plt.imshow(p,cmap = plt.cm.gray)
#     plt.savefig('C:/Users/DELL/Desktop/slike/'+str(br)+'polje12'+'.png')
#     br=br+1

print(polja.__len__())
# print(intersections)

plt.figure()
for i in intersections:
    plt.scatter(i[0][0], i[0][1])

# -----------------------predictions----------------------
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(5184, 700)
        self.lin2=nn.Linear(700,300)
        self.lin3=nn.Linear(300,13)

    def forward(self, xb):
        x=F.relu(self.lin(xb))
        x=F.relu(self.lin2(x))
        x=F.log_softmax(self.lin3(x),dim=1)
        return x

nnet=NN()
nnet.load_state_dict(torch.load("my_chess_model.pt"))
# for param in nnet.parameters():
#   print(param.data)
# nnet.eval()
#
# input_size = 5184  # 30x30  za svaku sliku
# hidden_sizes = [700, 200]
# output_size = 13  # 6 figura svake boje+prazno polje

transf = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize([30, 30]),
                             transforms.Grayscale(),
                             MyHOG(),
                             transforms.ToTensor(),

                             ])
switcher = {
    0: "bela kraljica",
    1: "beli konj",
    2: "beli kralj",
    3: "beli lovac",
    4: "beli pijun",
    5: "beli top",
    6: "crna kraljica",
    7: "crni konj",
    8: "crni kralj",
    9: "crni lovac",
    10: "crni pijun",
    11: "crni top",
    12: "prazno"
}


def switch(argument):
    return switcher[argument]

matrix=[]
br=0
for p in polja:
    # image processing openCV
    # dynamic thresholding instead of the statig one
    # _,p = cv2.threshold(p, 127, 255, cv2.THRESH_OTSU)
    # p = cv2.adaptiveThreshold(p, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20)
    # p=p/255.0

    # p = PIL.Image.fromarray(p)
    # test=p.numpy()[0]
    # plt.figure()
    # plt.imshow(p)

    # print(type(p))

    #p = Image.fromarray( p,'L')

    # plt.figure()
    # plt.imshow(p, )


    p = transf(p)

    p = p.view(1, 5184)

    with torch.no_grad():
        logps = nnet(p)
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    pred_label = probab.index(max(probab))

    plt.title(pred_label)

    #print(pred_label, switch(pred_label))
    matrix.append(switch(pred_label))

matrix1=np.reshape(matrix,(8,8),order='F')
# matrix2=np.asmatrix(matrix)

print(matrix1)
# print(matrix2)

plt.imshow(img)
plt.show()
