# -*- coding: utf-8 -*-
"""
Created on Thu May 01 11:00:56 2019

@author: sokil
"""

import numpy as np
import cv2

hsv = 0
hsv1=0
hsv2=0
team1_low1=0
team1_low2=0
team1_low3=0
team1_up1=0
team1_up2=0
team1_up3=0

team2_low1=0
team2_low2=0
team2_low3=0
team2_up1=0
team2_up2=0
team2_up3=0




def onMouse(event,x,y,flags,param):
    global hsv, hsv1, hsv2, team1_low1, team1_low2, team1_low3, team1_up1, team1_up2, team1_up3, team2_low1, team2_low2, team2_low3, team2_up1, team2_up2, team2_up3
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            print(CheckingTeam[y, x])
            team1 = CheckingTeam[y, x]
            
            first_pixel = np.uint8([[team1]])
            hsv = cv2.cvtColor(first_pixel, cv2.COLOR_BGR2HSV)
            
            hsv1=hsv[0][0]
            
            if hsv1[0] < 10:
                print("Team1_case1")
                team1_low1 = np.array([hsv[0]-10+180, 30, 30])
                team1_up1 = np.array([180, 255, 255])
                team1_low2 = np.array([0, 30, 30])
                team1_up2 = np.array([hsv[0], 255, 255])
                team1_low3 = np.array([hsv[0], 30, 30])
                team1_up3 = np.array([hsv[0]+10, 255, 255])
            
            elif hsv1[0] > 170:
                print("Team1_case2")
                team1_low1 = np.array([hsv[0], 30, 30])
                team1_up1 = np.array([180, 255, 255])
                team1_low2 = np.array([0, 30, 30])
                team1_up2 = np.array([hsv[0]+10-180, 255, 255])
                team1_low3 = np.array([hsv[0]-10, 30, 30])
                team1_up3 = np.array([hsv[0], 255, 255])
                
            else:
                print("Team1_case3")
                team1_low1 = np.array([hsv[0], 30, 30])
                team1_up1 = np.array([hsv[0]+10, 255, 255])
                team1_low2 = np.array([hsv[0]-10, 30, 30])
                team1_up2 = np.array([hsv[0], 255, 255])
                team1_low3 = np.array([hsv[0]-10, 30, 30])
                team1_up3 = np.array([hsv[0], 255, 255])
                
                
            print("Team1 hsv",hsv1)
            
            
            
    elif event == cv2.EVENT_RBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            print(CheckingTeam[y, x])
            team2 = CheckingTeam[y, x]
            
            second_pixel = np.uint8([[team2]])
            hsv = cv2.cvtColor(second_pixel, cv2.COLOR_BGR2HSV)
            hsv2=hsv[0][0]
            
            
            if hsv2[0] < 10:
                print("Team2_case1")
                team2_low1 = np.array([hsv[0]-10+180, 30, 30])
                team2_up1 = np.array([180, 255, 255])
                team2_low2 = np.array([0, 30, 30])
                team2_up2 = np.array([hsv[0], 255, 255])
                team2_low3 = np.array([hsv[0], 30, 30])
                team2_up3 = np.array([hsv[0]+10, 255, 255])
            
            elif hsv2[0] > 170:
                print("Team2_case2")
                team2_low1 = np.array([hsv[0], 30, 30])
                team2_up1 = np.array([180, 255, 255])
                team2_low2 = np.array([0, 30, 30])
                team2_up2 = np.array([hsv[0]+10-180, 255, 255])
                team2_low3 = np.array([hsv[0]-10, 30, 30])
                team2_up3 = np.array([hsv[0], 255, 255])
                
            else:
                print("Team2_case3")
                team2_low1 = np.array([hsv[0], 30, 30])
                team2_up1 = np.array([hsv[0]+10, 255, 255])
                team2_low2 = np.array([hsv[0]-10, 30, 30])
                team2_up2 = np.array([hsv[0], 255, 255])
                team2_low3 = np.array([hsv[0]-10, 30, 30])
                team2_up3 = np.array([hsv[0], 255, 255])
                
            print("Team2 hsv",hsv2)
            
cv2.namedWindow('CheckingTeam')
cv2.setMouseCallback('CheckingTeam',onMouse)


while(True):
    CheckingTeam = cv2.imread('20190413.png')
    height, width = CheckingTeam.shape[:2]
    #print("Size Checking",height,width)
    
    CheckingTeam = cv2.resize(CheckingTeam, (width, height), interpolation=cv2.INTER_AREA)
    
    img_hsv = cv2.cvtColor(CheckingTeam, cv2.COLOR_BGR2HSV)
    team1_low1_u=cv2.UMat(team1_low1)

    
    team1_mask1 = cv2.inRange(img_hsv, team1_low1, team1_up1)
    team1_mask2 = cv2.inRange(img_hsv, team1_low2, team1_up2)
    team1_mask3 = cv2.inRange(img_hsv, team1_low3, team1_up3)
    team2_mask1 = cv2.inRange(img_hsv, team2_low1, team2_up1)
    team2_mask2 = cv2.inRange(img_hsv, team2_low2, team2_up2)
    team2_mask3 = cv2.inRange(img_hsv, team2_low3, team2_up3)
    img_mask = team1_mask1 | team1_mask2 | team1_mask3 | team2_mask1 | team2_mask2 | team2_mask3


    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_result = cv2.bitwise_and(CheckingTeam, CheckingTeam, mask=img_mask)


    cv2.imshow('CheckingTeam', CheckingTeam)
    #cv2.imshow('img_hsv', img_hsv)
    cv2.imshow('img_result', img_result)


    # ESC 키누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break


cv2.destroyAllWindows()



