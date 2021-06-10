import cv2
import mediapipe as mp
import time


class FaceDetector():
    def __init__(self,minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetect = self.mpFaceDetection.FaceDetection()

    def findFaces(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.faceDetect.process(imgRGB)
        bboxs = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id,bbox,detection.score])
                if draw:
                    img = self.fancy_draw(img,bbox)
                    cv2.putText(img,"Accuracy: {}%".format(round(detection.score[0]*100,2)),(bbox[0],bbox[1]-20),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,0,0),2)
        return img, bboxs



    def fancy_draw(self,img,bbox,l=25,t=5,rt=1):
        x, y, w, h = bbox
        x1, y1 = x+w , y+h
        cv2.rectangle(img,bbox,(255,0,255),rt)
        #top left x,y
        cv2.line(img,(x,y),(x+l,y),(255,0,255),t)
        cv2.line(img,(x,y),(x,y+l),(255,0,255),t)
        #top right x1,y
        cv2.line(img,(x1,y),(x1-l,y),(255,0,255),t)
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),t)
        #bottom left x,y1
        cv2.line(img,(x,y1),(x+l,y1),(255,0,255),t)
        cv2.line(img,(x,y1),(x,y1-l),(255,0,255),t)
        #bottom right x1,y1
        cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),t)
        cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),t)

        return img

def main():
    capture = cv2.VideoCapture(0)
    pTime = 0
    detector = FaceDetector()

    while True:
        success,img = capture.read()
        img,bboxs = detector.findFaces(img)
        print(bboxs)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3)
        cv2.imshow("output",img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break



if __name__ == '__main__':
    main()