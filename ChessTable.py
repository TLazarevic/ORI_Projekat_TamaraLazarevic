import sys

import numpy as np
import cv2 # OpenCV biblioteka
import matplotlib
import matplotlib.pyplot as plt
import imutils

img = cv2.imread(r"C:\Users\DELL\Desktop\1.jpg")  # ucitavanje slike sa diska
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # konvertovanje iz BGR u RGB model boja (OpenCV ucita sliku kao BGR)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
maxy=0
miny=sys.maxsize
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
	if y1>maxy:
		maxy=y1
	if y2>maxy:
		maxy=y2
	if y2<miny:
		min=y2
	if y2 < miny:
			miny = y2
	cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

y=miny
x=0
h=maxy
w=1000
crop_img = img[y:h, x:x+w]

plt.imshow(crop_img)
plt.show()