import cv2
import numpy as np
import FaceDetectionModule as fdm



cap = cv2.VideoCapture(0)
cap.set(3,620)
cap.set(4,500)


detector = fdm.FaceDetector()

while True:
    suc,img = cap.read()
    face,bbox = detector.findFaces(img,draw=False)
    tempImg = img.copy()
    maskShape = (img.shape[0], img.shape[1], 1)
    mask = np.full(maskShape, 0, dtype=np.uint8)
    if len(bbox)>0:
        x,y,w,h = bbox[0][1]
        tempImg [y:y+h, x:x+w] = cv2.blur(tempImg [y:y+h, x:x+w] ,(33,33))
        cv2.circle(tempImg , ( int((x + x + w )/2), int((y + y + h)/2 )), int ((h / 2)+200), (0, 0, 0), 5)
        cv2.circle(mask , ( int((x + x + w )/2), int((y + y + h)/2 )), int (h / 2), (255), -1)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img,img,mask = mask_inv)
    img2_fg = cv2.bitwise_and(tempImg,tempImg,mask = mask)
    dst = cv2.add(img1_bg,img2_fg)
    cv2.imshow("Output",dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break