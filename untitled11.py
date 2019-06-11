# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:07:32 2019

@author: sokil
"""

import cv2
import numpy as np

cap=cv2.VideoCapture('2019_03_28_11_17_25_745.mp4')
if(not cap.isOpened()):
    print('Error opening video')
    
height, width=(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
               int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

TH=40
AREA_TH=80
acc_bgr=np.zeros(shape=(height,width,3), dtype=np.float32)

mode=cv2.RETR_EXTERNAL
method=cv2.CHAIN_APPROX_SIMPLE

t=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    t+=1
    print('t=',t)
    blur=cv2.GaussianBlur(frame, (5,5),0.0)
    
    if t<50:
        cv2.accumulate(blur, acc_bgr)
        continue
    elif t==50:
        bkg_bgr=acc_bgr/t
    elif t>50:
        #diff_bgr=cv2.absdiff(np.float32(blur),bkg_bgr).astype(np.uint8)
        diff_bgr=np.uint8(cv2.absdiff(np.float32(blur),bkg_bgr))
        db,dg,dr=cv2.split(diff_bgr)
        ret,bb=cv2.threshold(db,TH,255,cv2.THRESH_BINARY)
        ret,bg=cv2.threshold(dg,TH,255,cv2.THRESH_BINARY)
        ret,br=cv2.threshold(dr,TH,255,cv2.THRESH_BINARY)
        
        blmage=cv2.bitwise_or(bb,bg)
        blmage=cv2.bitwise_or(br,blmage)
        
        blmage=cv2.erode(blmage,None,5)
        blmage=cv2.dilate(blmage,None,5)
        blmage=cv2.erode(blmage,None,7)
        
        cv2.imshow('blmage',blmage)
        msk=blmage.copy()
        contours, hierachy = cv2.findContours(blmage, mode, method)
        cv2.drawContours(frame, contours,-1,(255,0,0),1)
        
        for i, cnt in enumerate(contours):
            area=cv2.contourArea(cnt)
            if area>AREA_TH:
                x,y,width,height=cv2.boundingRect(cnt)
                cv2.rectangle(frame,(x,y),(x+width, y+height),(0,0,255),2)
                cv2.rectangle(msk,(x,y),(x+width, y+height),255,-1)
            
        msk=cv2.bitwise_not(msk)
        cv2.accumulateWeighted(blur, bkg_bgr, alpha=0.1, mask=msk)
        
        cv2.imshow('frame',frame)
        cv2.imshow('bkg_bgr',np.uint8(bkg_bgr))
        cv2.imshow('diff_bgr',diff_bgr)
        
        key=cv2.waitKey(25)
        if key==27:
            break
            
if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()