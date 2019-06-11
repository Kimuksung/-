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

line_low1=0
line_up1=0
line_low2=0
line_up2=0
line_low3=0
line_up3=0

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
        
cv2.namedWindow('live')
cv2.setMouseCallback('live',check)

capture=cv2.VideoCapture('offsidetrap.mp4')
cv2.namedWindow('live')
colors=[tuple(255*np.random.rand(3)) for i in range(5)]

while (capture.isOpened()):
    
    if cv2.waitKey(1)==32:
        decise_v+=1
        
    while(decise_v%2==1):
        #print('pause')
        if cv2.waitKey(1)==32:
            decise_v+=1
    
    stime=time.time()
    ret, frame =capture.read()
    results=tfnet.return_predict(frame)
    
    output=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    image_mask1=cv2.inRange(image_hsv, line_low1, line_up1)
    image_mask2=cv2.inRange(image_hsv, line_low2, line_up2)
    image_mask3=cv2.inRange(image_hsv, line_low3, line_up3)
    image_mask=image_mask1|image_mask2|image_mask3
    
    team1_mask1=cv2.inRange(image_hsv, team1_low1, team1_up1)
    team1_mask2=cv2.inRange(image_hsv, team1_low2, team1_up2)
    team1_mask3=cv2.inRange(image_hsv, team1_low3, team1_up3)
    team1_mask=team1_mask1|team1_mask2|team1_mask3
    
    team2_mask1=cv2.inRange(image_hsv, team2_low1, team2_up1)
    team2_mask2=cv2.inRange(image_hsv, team2_low2, team2_up2)
    team2_mask3=cv2.inRange(image_hsv, team2_low3, team2_up3)
    team2_mask=team2_mask1|team2_mask2|team2_mask3
    
    if ret:
        for color, result in zip(colors, results):
            tl=(result['topleft']['x'], result['topleft']['y'])
            br=(result['bottomright']['x'],result['bottomright']['y'])
            label=result['label']
            
            if label=='person':
                if team1_mask.any():
                    label='Team1'
                elif team2_mask.any():
                    label='Team2'
            
            frame=cv2.rectangle(frame, tl,br,color,7)
            frame=cv2.putText(frame, label,tl,cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
        
        cv2.imshow('live',frame)
        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break
    else:
        capture.release
        cv2.destroyAllWindows()
        break