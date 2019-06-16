import sys

import numpy as np
import cv2 # OpenCV biblioteka
import matplotlib
import matplotlib.pyplot as plt
import imutils

from tkinter import filedialog

#--------------otvaranje slike-----------------

file_path = filedialog.askopenfilename()

img = cv2.imread(file_path)  # ucitavanje slike sa diska
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # konvertovanje iz BGR u RGB model boja (OpenCV ucita sliku kao BGR)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#--------pronalazenje linija na slici table----------------

edges = cv2.Canny(gray,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,320)

# maxy=0
# miny=sys.maxsize

lines=np.unique(lines,axis=0)

for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # if y1>maxy:
    # 	maxy=y1
    # if y2>maxy:
    # 	maxy=y2
    # if y2<miny:
    # 	min=y2
    #if y1 < miny:
    #		miny = y1

#------------------pronalazenje preseka linija---------------------

vertlines=lines[np.where(lines[:,:,1]==np.pi/2)]
horlines=lines[np.where(lines[:,:,1]==0)]


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations."""

    rho1=line1[0]
    theta1 = line1[1]
    rho2=line2[0]
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

#-------brisanje duplih preseka-----

intersections.sort()

icopy = intersections;

for i in range((icopy.__len__()-1),-1,-1):
    if( (intersections[i][0][1]-intersections[i-1][0][1])<2.5):
        intersections.remove(intersections[i])

#-----sredjivane nedetektovanih ivica slike----------------

fieldLen=intersections[1][0][1]-intersections[0][0][1]

x=[]
for i in intersections:
    x.append(i[0][0])
x=np.unique(x)

if(intersections[0][0][1]-fieldLen) in range(-5,5): #fali gornja ivica
    for i in x:
        intersections.append([[i,2]])

if(intersections[-1][0][1]+fieldLen) in range(-5+img.shape[1],5+img.shape[1]): #fali donja ivica
    for i in x:
        # if(intersections[-1][0][1]+fieldLen)>img.shape[1]:
        intersections.append([[i, img.shape[1]-2]])

intersections.sort()
print(intersections.__len__())
print(intersections)


for i in intersections:
    plt.scatter(i[0][0],i[0][1])

plt.imshow(img)
plt.show()