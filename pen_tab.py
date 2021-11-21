import cv2 as cv
import numpy as np
import mouse as mp
cam=cv.VideoCapture(0)
while cam.isOpened():
    ret,frame=cam.read()
    frame=cv.resize(frame,(1596,892))
    #'purple': [[158, 255, 255], [129, 50, 70]]
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    lower = np.array([129, 50, 70])
    upper = np.array([158, 255, 255])
    mask=cv.inRange(hsv,lower,upper)
    #getting contours
    contours,_=cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    count_cnt=0
    cnt_array=[]
    max_area=0
    for cnt in contours:
        area=cv.contourArea(cnt)

        x,y,w,h=cv.boundingRect(cnt)

        # cv.drawContours(frame, cnt, -1, (0, 255, 0), 2)
        max_area=max(max_area,area)
        cnt_array.append([count_cnt,area, cnt])
        count_cnt += 1
    for id,area,cnt in cnt_array:
        if area==max_area and area!=0:
            # cv.putText(frame, str(count_cnt), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0))
            M = cv.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv.circle(frame, (cX, cY), 10, (255, 0, 0), 2)

            mp.move(cX,cY)
            break;

    cv.imshow("video", frame)
    cv.imshow("gray", mask)

    if cv.waitKey(25) & 0xFF==ord("q"):
        cam.release()
        break
# x=1596
#y=892

cam.release()
cv.destroyAllWindows()
cv.waitKey(0)
