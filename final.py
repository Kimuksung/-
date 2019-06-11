# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 20:10:12 2019

@author: sokil
"""

import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option={
        'model':'cfg/yolo.cfg',
        'load':'bin/yolo.weights',
        'threshold': 0.1,
        'gpu':1.0
}

tfnet=TFNet(option)

decise_v=0

line_low1=np.array([30,40,110])
line_up1=np.array([50,100,180])
line_low2=np.array([30,40,110])
line_up2=np.array([50,100,180])
line_low3=np.array([30,40,110])
line_up3=np.array([50,100,180])

team1_low1=0
team1_up1=0
team1_low2=0
team1_up2=0
team1_low3=0
team1_up3=0

team2_low1=0
team2_up1=0
team2_low2=0
team2_up2=0
team2_low3=0
team2_up3=0

def check(event,x,y,flags,param):
    global line_low1, line_up1, line_low2, line_up2, line_low3, line_up3, team1_low1, team1_up1, team1_low2, team1_up2, team1_low3, team1_up3, team2_low1, team2_up1, team2_low2, team2_up2, team2_low3, team2_up3
    if event == cv2.EVENT_LBUTTONDOWN:
        #print('check')
        #print(frame[y,x])
        line = frame[y, x]
        line_pixel = np.uint8([[line]])
        
        hsv=cv2.cvtColor(line_pixel, cv2.COLOR_BGR2HSV)
        h=hsv[0][0]
        
        if h[0]<10:
            line_low1 = np.array([h[0]-10+180, 30, 30])
            line_up1 = np.array([179, 255, 255])
            line_low2 = np.array([0, 30, 30])
            line_up2 = np.array([h[0], 255, 255])
            line_low3 = np.array([h[0], 30, 30])
            line_up3 = np.array([h[0]+10, 255, 255])

        
        elif h[0] > 169:
            line_low1 = np.array([h[0], 30, 30])
            line_up1 = np.array([179, 255, 255])
            line_low2 = np.array([0, 30, 30])
            line_up2 = np.array([h[0]+10-180, 255, 255])
            line_low3 = np.array([h[0]-10, 30, 30])
            line_up3 = np.array([h[0], 255, 255])
        
        else:
            line_low1 = np.array([h[0], 30, 30])
            line_up1 = np.array([h[0]+10, 255, 255])
            line_low2 = np.array([h[0]-10, 30, 30])
            line_up2 = np.array([h[0], 255, 255])
            line_low3 = np.array([h[0]-10, 30, 30])
            line_up3 = np.array([h[0], 255, 255])
     
    if event == cv2.EVENT_RBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            team1 = frame[y,x]
            
            team1_pixel=np.uint8([[team1]])
            hsv=cv2.cvtColor(team1_pixel, cv2.COLOR_BGR2HSV)
            h=hsv[0][0]
            
            if h[0]<15:
                team1_low1 = np.array([h[0]-15+180, 30, 30])
                team1_up1 = np.array([179, 255, 255])
                team1_low2 = np.array([0, 30, 30])
                team1_up2 = np.array([h[0], 255, 255])
                team1_low3 = np.array([h[0], 30, 30])
                team1_up3 = np.array([h[0]+15, 255, 255])
            
            elif h[0] > 164:
                team1_low1 = np.array([h[0], 30, 30])
                team1_up1 = np.array([179, 255, 255])
                team1_low2 = np.array([0, 30, 30])
                team1_up2 = np.array([h[0]+15-180, 255, 255])
                team1_low3 = np.array([h[0]-15, 30, 30])
                team1_up3 = np.array([h[0], 255, 255])
            
            else:
                team1_low1 = np.array([h[0], 30, 30])
                team1_up1 = np.array([h[0]+10, 255, 255])
                team1_low2 = np.array([h[0]-10, 30, 30])
                team1_up2 = np.array([h[0], 255, 255])
                team1_low3 = np.array([h[0]-10, 30, 30])
                team1_up3 = np.array([h[0], 255, 255])
            
    if event == cv2.EVENT_RBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            team2 = frame[y,x]
            
            team2_pixel=np.uint8([[team2]])
            hsv=cv2.cvtColor(team2_pixel, cv2.COLOR_BGR2HSV)
            h=hsv[0][0]
            
            if h[0]<15:
                team2_low1 = np.array([h[0]-15+180, 30, 30])
                team2_up1 = np.array([179, 255, 255])
                team2_low2 = np.array([0, 30, 30])
                team2_up2 = np.array([h[0], 255, 255])
                team2_low3 = np.array([h[0], 30, 30])
                team2_up3 = np.array([h[0]+15, 255, 255])
            
            elif h[0] > 164:
                team2_low1 = np.array([h[0], 30, 30])
                team2_up1 = np.array([179, 255, 255])
                team2_low2 = np.array([0, 30, 30])
                team2_up2 = np.array([h[0]+15-180, 255, 255])
                team2_low3 = np.array([h[0]-15, 30, 30])
                team2_up3 = np.array([h[0], 255, 255])
            
            else:
                team2_low1 = np.array([h[0], 30, 30])
                team2_up1 = np.array([h[0]+10, 255, 255])
                team2_low2 = np.array([h[0]-10, 30, 30])
                team2_up2 = np.array([h[0], 255, 255])
                team2_low3 = np.array([h[0]-10, 30, 30])
                team2_up3 = np.array([h[0], 255, 255])

cv2.namedWindow('frame')
cv2.setMouseCallback('frame',check)

def definePass(img, flow, thresh=2, stride=8):
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
    print(df_pass)

average_inclination=0.0
team_a_x=[(9999,9999,9999)]
team_b_x=[(9999,9999,9999)]

font = cv2.FONT_HERSHEY_SIMPLEX
#lower_white = np.array([30,40,110])
#upper_white = np.array([50,100,180])

#lower_a = np.array([7,175,120])
#upper_a = np.array([23,255,205])
#lower_b = np.array([106,30,30])
#upper_b = np.array([118,175,180])

capture=cv2.VideoCapture('soccer.mp4')
cv2.namedWindow('frame')
colors=[tuple(255*np.random.rand(3)) for i in range(5)]

height, width=(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
               int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))

ret, frame=capture.read()
imgP=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

TH=2
AREA_TH=50
mode=cv2.RETR_EXTERNAL
method=cv2.CHAIN_APPROX_SIMPLE
params=dict(pyr_scale=0.3,levels=3,winsize=3,iterations=3,poly_n=5,poly_sigma=1.1,flags=0)

while (capture.isOpened()):
    if cv2.waitKey(1)==32:
        decise_v+=1
    
    while(decise_v%2==1):
        #print('pause')
        if cv2.waitKey(1)==32:
            decise_v+=1
    stime=time.time()
    ret, frame =capture.read()
    
    imgC=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    imgC=cv2.GaussianBlur(imgC,(5,5),0.5)
    
    flow=cv2.calcOpticalFlowFarneback(imgP,imgC,None,**params)
    definePass(frame,flow,TH)
    
    imgP=imgC.copy()
    
    max_x,max_y,t = frame.shape
    image_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    #hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    #hist2 = cv2.calcHist([hsv], [1, 2], None, [256, 256], [0, 256, 0, 256])
    #mask_white=cv2.inRange(hsv,lower_white,upper_white)
    
    image_mask1=cv2.inRange(image_hsv, line_low1, line_up1)
    image_mask2=cv2.inRange(image_hsv, line_low2, line_up2)
    image_mask3=cv2.inRange(image_hsv, line_low3, line_up3)
    image_mask=image_mask1|image_mask2|image_mask3
    
    result2 = cv2.bitwise_and(frame,frame,mask=image_mask)
    
    gray = cv2.cvtColor(result2,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150)
    
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    i=1
    inclination_arr=[]
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
        inclination = (x2-x1)/(y2-y1)
        
        if(-5<inclination<5):
            inclination_arr.append(inclination)
    temp=0.0
    for a in inclination_arr:
        temp=temp+a
    if(len(inclination_arr)!=0):
        average_inclination = temp/len(inclination_arr)
    else:
        inclination_arr=0
    
    #yolo detect
    result = tfnet.return_predict(frame)
    
    for c in result:
        x=c['topleft']['x']
        y=c['topleft']['y']
        w=c['bottomright']['x']-x
        h=c['bottomright']['y']-y
        
        if(w<100):
            player_img = frame[y:y+h,x:x+w]
            player_hsv = cv2.cvtColor(player_img,cv2.COLOR_BGR2HSV)
            
            team1_mask1=cv2.inRange(player_hsv, team1_low1, team1_up1)
            team1_mask2=cv2.inRange(player_hsv, team1_low2, team1_up2)
            team1_mask3=cv2.inRange(player_hsv, team1_low3, team1_up3)
            team1_mask=team1_mask1|team1_mask2|team1_mask3
            
            #mask1 = cv2.inRange(player_hsv, lower_a, upper_a)
            res1 = cv2.bitwise_and(player_img, player_img, mask=team1_mask)
            res1 = cv2.cvtColor(res1,cv2.COLOR_HSV2BGR)
            res1 = cv2.cvtColor(res1,cv2.COLOR_BGR2GRAY)
            ACount = cv2.countNonZero(res1)
            
            if(ACount >= 30):
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                cv2.putText(frame, 'A', (x-2, y-2), font, 0.8, (255,0,0), 2, cv2.LINE_AA)
                if(average_inclination<0):
                    temp = int(y-x/average_inclination)
                    team_a_x.append((x,y,temp))
                elif(average_inclination>0):
                    temp = int(y-x/average_inclination)
                    team_a_x.append((x+w,y,temp))
                else:
                    print('no inclination')
            
            team2_mask1=cv2.inRange(player_hsv, team2_low1, team2_up1)
            team2_mask2=cv2.inRange(player_hsv, team2_low2, team2_up2)
            team2_mask3=cv2.inRange(player_hsv, team2_low3, team2_up3)
            team2_mask=team2_mask1|team2_mask2|team2_mask3
            
            #mask2 = cv2.inRange(player_hsv, lower_b, upper_b)
            res2 = cv2.bitwise_and(player_img, player_img, mask=team2_mask)
            res2 = cv2.cvtColor(res2,cv2.COLOR_HSV2BGR)
            res2 = cv2.cvtColor(res2,cv2.COLOR_BGR2GRAY)
            BCount = cv2.countNonZero(res2)
            
            if(BCount >=30):
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),3)
                cv2.putText(frame, 'B', (x-2, y-2), font, 0.8, (255,255,0), 2, cv2.LINE_AA)
                if(average_inclination<0):
                    temp = int(y-x/average_inclination)
                    team_b_x.append((x,y,temp))
                elif(average_inclination>0):
                    temp = int(y-x/average_inclination)
                    team_b_x.append((x+w,y,temp))
                else:
                    print('no inclination')
                    
    if(average_inclination<0):                
        team_a_x.sort(key=lambda tup: tup[2])
        team_b_x.sort(key=lambda tup: tup[2])
        
        cv2.line(frame,(int(-team_a_x[0][2]*average_inclination),0),(0,int(team_a_x[0][2])),(0,0,255),2)
        cv2.line(frame,(int(-team_b_x[0][2]*average_inclination),0),(0,int(team_b_x[0][2])),(30,255,255),2)
    
    else:
        team_a_x.sort(key=lambda tup: tup[2])
        team_b_x.sort(key=lambda tup: tup[2])
        
        cv2.line(frame,(int((max_y-int(team_a_x[0][2]))*average_inclination),max_y),(0,int(team_a_x[0][2])),(0,0,255),2)
        cv2.line(frame,(int((max_y-int(team_b_x[0][2]))*average_inclination),max_y),(0,int(team_b_x[0][2])),(30,255,255),2)
    
    cv2.imshow('frame',frame)
    imgP=imgC.copy()
    key=cv2.waitKey(25)
    if key==27:
        break
if capture.isOpened():
    capture.release();
cv2.destroyAllWindows()