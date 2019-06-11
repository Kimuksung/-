# -*- coding: utf-8 -*-
"""
Created on Thu April 30 10:54:41 2019

@author: sokil
"""

import numpy as np
import cv2


def hsv():
    blue=np.uint8([[[255,0,0]]])
    green=np.uint8([[[0,255,0]]])
    red=np.uint8([[[0,0,255]]])
    
    hsv_blue=cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
    hsv_green=cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    hsv_red=cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
    
    print('blue : ',hsv_blue)
    print('green : ',hsv_green)
    print('red : ',hsv_red)
    
hsv()