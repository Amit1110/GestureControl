import cv2
import numpy as np
import math

def main():
    cap = cv2.VideoCapture(0)
    
    cap.set(3,5000)
    cap.set(4,5000)
    
    print(str(cap.get(3)))
    print(str(cap.get(4)))
    
    if cap.isOpened():
        ret, frame = cap.read()
    else:
        ret = False
    
    while ret is True:
        ret, frame = cap.read()
        
        frame = cv2.flip(frame,1)
        
        #cv2.circle(frame, (1270,710),10,(0,0,255),-1)
        
        cv2.rectangle(frame,(800,0),(1280,500),(0,0,255),1)
        
        hand = frame[0:500,800:1280]
        
        grey = cv2.cvtColor(hand,cv2.COLOR_BGR2GRAY)
        
        #kernel = np.ones((5,5),np.float32)/25
        #smooth = cv2.filter2D(hand,-1,kernel)
        
        blur = cv2.GaussianBlur(grey,(25,25),0)
        
        r, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        cnt = max(contours, key = lambda x: cv2.contourArea(x))
        
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(hand,(x,y),(x+w,y+h),(0,255,255),0)
        
        hull = cv2.convexHull(cnt)

        black = np.zeros(hand.shape,np.uint8)
        cv2.drawContours(black, [cnt], 0, (0, 255, 0), 0)
        cv2.drawContours(black, [hull], 0,(0, 0, 255), 0)
        count_defects = 0
        
        hull = cv2.convexHull(cnt, returnPoints=False)
        
        #cv2.drawContours(frame[0:500,800:1280], contours, -1, (0,255,0), 3)
        
        defects = cv2.convexityDefects(cnt,hull)
        
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
    
            
            if angle <= 90:
                count_defects += 1
                cv2.circle(hand, far, 5, [0,0,255], -1)
        
            cv2.line(hand,start, end, [0,255,0], 2)
        
        print(count_defects)
            
        
        
    
    
            
        
        cv2.imshow('thresh',thresh)
        cv2.imshow('im2',im2)
        cv2.imshow('hand',hand)
        cv2.imshow('black',black)
        cv2.imshow('live video',frame)
        
        
        if cv2.waitKey(1) == 27:
            break
        
        
    cv2.destroyAllWindows()
    cap.release()


if __name__=="__main__":
    main()