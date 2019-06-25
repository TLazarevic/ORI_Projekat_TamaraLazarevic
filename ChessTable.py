import sys
from collections import Counter
from dataclasses import field

import numpy as np
import cv2  # OpenCV biblioteka
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


from tkinter import filedialog

# --------------otvaranje slike-----------------

file_path = filedialog.askopenfilename()

img = cv2.imread(file_path)  # ucitavanje slike sa diska
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # konvertovanje iz BGR u RGB model boja (OpenCV ucita sliku kao BGR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --------pronalazenje linija na slici table----------------

edges = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 280)

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
#
# intersections=sorted(intersections, key=lambda coor: coor[0][0])
# print("intersections: ",intersections.__len__())
# print("intersections: ",intersections)
#
#
icopy = intersections;

print("intersections: ",intersections.__len__())
print("intersections: ",intersections)

for i in range((icopy.__len__() - 1), 0, -1):
    if ((intersections[i][0][0] - intersections[i - 1][0][0]) < 5) and (intersections[i][0][1] in range (intersections[i - 1][0][1]-5,intersections[i - 1][0][1]+5)) :
        intersections.remove(intersections[i])

print("intersections: ",intersections.__len__())
print("intersections: ",intersections)

intersections=sorted(intersections, key=lambda coor: coor[0][0])
print("intersections: ",intersections.__len__())
print("intersections: ",intersections)

for i in range((icopy.__len__() - 1), 0, -1):
    if ((intersections[i][0][1] - intersections[i - 1][0][1]) < 5) and (intersections[i][0][0] in range(intersections[i - 1][0][0]-5,intersections[i - 1][0][0]+5)):
        intersections.remove(intersections[i])

print("intersections: ",intersections.__len__())
print("intersections: ",intersections)


# ------------------dodavanje nedetektovanih ivica slike----------------

intersections.sort()
distancesW = []
distancesL = []

for i in range(0, intersections.__len__() - 1):
    distancesL.append(intersections[i + 1][0][1] - intersections[i][0][1])
    distancesW.append(intersections[i + 1][0][0] - intersections[i][0][0])

if ((Counter(distancesL).most_common(1)[0][0]) != 0):
    fieldLen = Counter(distancesL).most_common(1)[0][0]
else:
    fieldLen = Counter(distancesL).most_common(2)[1][0]

if ((Counter(distancesW).most_common(1)[0][0]) != 0):
    fieldWid = Counter(distancesW).most_common(1)[0][0]
else:
    fieldWid = Counter(distancesW).most_common(2)[1][0]

x = []
for i in intersections:
    x.append(i[0][0])
x = np.unique(x)

firstYrowdot = intersections[0][0][1] - fieldLen  # adding this as Y coord to top row instead of zero to make space to make algorithm more flexible
if firstYrowdot < 0:
    while firstYrowdot < 0:
        firstYrowdot = firstYrowdot + 1


lastYrowdot = intersections[-1][0][1] + fieldLen

if lastYrowdot > img.shape[0]:
    while lastYrowdot > img.shape[0]:
        lastYrowdot = lastYrowdot - 1

if (intersections[-1][0][1] + fieldLen) in range(-4 + img.shape[0], img.shape[0] + 4):  # fali donja ivica
    for i in x:
        intersections.append([[i, lastYrowdot]])
        # intersections.append([[i, img.shape[1]-2]])

if (intersections[0][0][1] - fieldLen) in range(-4, 4):  # fali gornja ivica
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

print(lastXrowdot, intersections[-1][0][0], intersections[-1][0][0] + fieldWid, img.shape[0])

if (intersections[-1][0][0] + fieldWid) in range(-4 + img.shape[1], img.shape[1] + 4):  # fali desna ivica
    for i in y:
        intersections.append([[lastXrowdot, i]])
        # intersections.append([[i, img.shape[0]-2]])

if (intersections[0][0][0] - fieldWid) in range(-4, 4):  # fali leva ivica
    for i in y:
        # intersections.append([[i,2]])
        intersections.append([[firstXrowdot, i]])

#----------------------pronalazenje i filtriranje polja------------------------

sortx = sorted(intersections)
sorty = sorted(intersections, key=lambda coor: coor[0][1])

polja = []
velicinepolja = []

for i in range(0, intersections.__len__() - 1):
    for j in range(0, intersections.__len__() - 1):
        if img[sorty[j][0][1]:sorty[j + 1][0][1], sortx[i][0][0]:sortx[i + 1][0][0]].size != 0:
            print((sorty[j + 1][0][1] - sorty[j][0][1]) - (sortx[i + 1][0][0] - sortx[i][0][0]))
            if ((sorty[j + 1][0][1] - sorty[j][0][1]) - (sortx[i + 1][0][0] - sortx[i][0][0])) in range(-20,20):  # square check
                polja.append(img[sorty[j][0][1]:sorty[j + 1][0][1], sortx[i][0][0]:sortx[i + 1][0][0]])
                velicinepolja.append(polja[-1].size)

velicina = Counter(velicinepolja).most_common(1)[0][0]

temp = polja.__len__()
print(polja.__len__())

for p in range(temp - 1, 0, -1):
    if not (polja[p].size in range(velicina - 7000, velicina + 7000)):
        polja.pop(p)
    else:
        print(polja[p].size,velicina)
        polja[p] = cv2.resize(polja[p], dsize=(30, 30), interpolation=cv2.INTER_CUBIC)
        polja[p] = cv2.cvtColor(polja[p], cv2.COLOR_BGR2GRAY)


#------------------------plots---------------------------------

for p in polja:
    plt.figure()
    plt.imshow(p)

print(polja.__len__())
# print(intersections)

plt.figure()
for i in intersections:
    plt.scatter(i[0][0], i[0][1])


plt.imshow(img)
plt.show()


