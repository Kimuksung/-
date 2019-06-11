# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:11:49 2019

@author: sokil
"""

import cv2
import numpy as np


def drawFlow(img, flow, thresh=2, stride=8):
    h, w=img.shape[:2]
    mag, ang=cv2.cartToPolar(flow[...,0],flow[...,1])
    flow2=np.int32(flow)
    df_pass=0
    for y in range(0,h,stride):
        for x in range(0,w,stride):
            dx,dy=flow2[y,x]
            diff_x=abs(dx)
            diff_y=abs(dy)
            
            diff=diff_x+diff_y
            if df_pass<diff :
                df_pass=diff
            
            #if mag[y,x]>thresh:
                #cv2.circle(img,(x,y),2,(0,255,0),-1)
                #cv2.line(img,(x,y),(x+dx,y+dy),(255,0,0),1)
    #print(df_pass)
    if (df_pass>100):
        print('pass')
    
cap=cv2.VideoCapture('soccer.mp4')
if(not cap.isOpened()):
    print('Error opening video')
height, width=(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
               int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
hsv=np.zeros((height,width,3),dtype=np.uint8)

ret, frame=cap.read()
imgP=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

TH=2
AREA_TH=50
mode=cv2.RETR_EXTERNAL
method=cv2.CHAIN_APPROX_SIMPLE
params=dict(pyr_scale=0.3,levels=3,winsize=3,iterations=3,poly_n=5,poly_sigma=1.1,flags=0)

t=0
while True:
    ret,frame=cap.read()
    if not ret:break
    #t+=1
    #print('t=',t)
    imgC=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    imgC=cv2.GaussianBlur(imgC,(5,5),0.5)
    
    flow=cv2.calcOpticalFlowFarneback(imgP,imgC,None,**params)
    drawFlow(frame,flow,TH)
    
            
    cv2.imshow('frame',frame)
    imgP=imgC.copy()
    key=cv2.waitKey(25)
    if key==27:
        break
if cap.isOpened():
    cap.release();
cv2.destroyAllWindows()