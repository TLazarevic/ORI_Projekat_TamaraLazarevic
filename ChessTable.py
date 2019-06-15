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
lines = cv2.HoughLines(edges,1,np.pi/180,350)

# maxy=0
# miny=sys.maxsize


# for i in range (len(lines)-1,0,-1):
#     print(lines[i,0])
#     if ((lines[i,0,0]==lines[i-1,0,0]) and (lines[i,0,1]==lines[i-1,0,1])):
#         lines=np.delete(lines,i,0)

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


#------------provera duplikata----------

lines=np.sort(lines,0)

print(lines)


print(len(lines))




print(lines)

plt.imshow(img)
plt.show()