# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:09:14 2019

@author: kimuk
"""

import cv2
 
# 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
vidcap = cv2.VideoCapture('offside.mp4')
 

while(vidcap.isOpened()):
    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
    ret, image = vidcap.read()
    cv2.imshow('original',image)
    # 캡쳐된 이미지를 저장하는 함수 

    if cv2.waitKey(1) & 0xFF == 27:
        cv2.imwrite('img2.png' , image)
        print('image 저장')

 
 
vidcap.release()