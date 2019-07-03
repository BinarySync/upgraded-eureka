# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import cv2

proj_dir = "C:/NeoTokyo/Documents/GitHub/upgraded-eureka/codes/"


#########[CAMERA]#########
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("Could not open video device")
# Read picture. ret === True on success
ret, frame = video_capture.read()
# Close device
video_capture.release()
###########################

#######[IMG IMPORT]########
#img = cv2.imread("D:/image.jpg")
img = cv2.imread("C:/Users/Acer/Pictures/Capturar.png")
#Usando dois argumentos, o segundo trata de 
#uma chave para decidir usar preto e branco
#img = cv2.imread("D:/img.png",0)
############################

######[IMG RESIZING]########
size = 2;
resized = cv2.resize(img,   (int(img.shape[1]*size) , int(img.shape[0]*size))   )
############################

########[IMG INFO]##########
#print(img)
#print(type(img))
#print(img.shape)
############################

#######[IMG SHOWING]########
#cv2.imshow("Window",resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
############################

######[IMG WITH TEXT]#######
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

cv2.putText(img,'Hello World!', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    lineType)
#############################


#TESTING
import cv2
##QUICK BOOT
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("Could not open video device")
ret, frame = video_capture.read()
video_capture.release()

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

proj_dir = "C:/NeoTokyo/Documents/GitHub/upgraded-eureka/codes/"
face_cascade = cv2.CascadeClassifier(proj_dir+"haarcascade_frontalface_default.xml")

##LOOP
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    raise Exception("Could not open video device")
loops = 0;
while (ret == True and loops<500):
    ret, frame = video_capture.read()
    #IMG PROCESSING    
    cv2.putText(frame,'frame:'+str(loops), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    #Face recognition
    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor=1.2,
        minNeighbors=5
        )
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Camera Frame",frame)
    cv2.waitKey(1)
    loops = loops + 1
# Close device
cv2.destroyAllWindows()
video_capture.release()
#FIM TESTE

#rec = cv2.face.EigenFaceRecognizer_create()
#rec = cv2.face.FisherFaceRecognizer_create()
#rec = cv2.face.LBPHFaceRecognizer_create()

face_cascade = cv2.CascadeClassifier(proj_dir+"haarcascade_frontalface_default.xml")

gray_img = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=1.2,
        minNeighbors=5
        )

for (x, y, w, h) in faces:
    cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Window",gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
