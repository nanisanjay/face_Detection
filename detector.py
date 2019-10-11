import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainingdata.yml")
id=0
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
thickness=2
fontcolor = (0,0,255) 
while(True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),3)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if(id==1):
            id="sanjay"
        elif(id==2):
            id="sushmitha"
        elif(id==3):
            id="apj"
        elif(id==4):
            id="gautam sir"
        elif(id==5):
            id="amman"
        elif(id==6):
            id="jk"
        else:
            id="none"
        cv2.putText(img,str(id),(x,y+h),font,1,fontcolor,thickness);
    cv2.imshow("FaceDetection",img);
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release()
cv2.destroyAllWindows()
