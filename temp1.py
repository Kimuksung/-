# -*- coding: utf-8 -*-
"""
Created on Wed May 29 00:07:48 2019

@author: kimmi
"""


import cv2
from matplotlib import pyplot as plt

img =cv2.imread('GB.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
x=hsv.shape[0]
y=hsv.shape[1]
L=[]
h=[0]*256
x1=0
y1=0
x2=0
y2=0
def mouse_callback(event,x,y,flags,param):
    global x1,y1,x2,y2
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Button down event x: '+str(x)+'y: '+str(y))
        x1=x
        y1=y
    elif event == cv2.EVENT_LBUTTONUP:
        print('Button up event x: '+str(x)+'y: '+str(y))
        x2=x
        y2=y

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)

while(1):

    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if(x2 != 0 and y2!=0):
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),3)
    if k == 27:
        print( "ESC 키 눌러짐")
        break

cv2.destroyAllWindows()  
print('x1: '+str(x1)+'y1: '+str(y1) +'x2 :' +str(x2)+'y2:'+str(y2))

for i in range(x1,x2):
    for j in range(y1,y2):
        L.append(hsv[j][i][1])
        
for i in range(len(L)):
    h[L[i]] = h[L[i]]+1   

for i in range(len(h)):
    print(str(i)+":"+str(h[i]))

plt.hist(L,100,[0,256])
plt.show()