# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:00:45 2020

@author: fewii
"""

import cv2

#proj_dir = "C:/NeoTokyo/Documents/GitHub/upgraded-eureka/codes/"
proj_dir = "D:/Git/upgraded-eureka/codes/"

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
#img = cv2.imread("C:/Users/Acer/Pictures/Capturar.png")
#Usando dois argumentos, o segundo trata de 
#uma chave para decidir usar preto e branco
#Aparentemente, tanto imread quanto o np.array servem para importar a imagem
img = cv2.imread("C:/Users/Acer/Pictures/Capturar.png",0)

import numpy as np
from PIL import Image
#Usar o Convert 'L' transforma a imagem em grayscale
faceImg = Image.open("C:/Users/Acer/Pictures/Capturar.png").convert('L')
faceNp = np.array(faceImg,'uint8')

cv2.imshow("Test",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Test",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
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

##########[GETTING CODE DIR]##################
import os
print(os.path)
dirpath = os.getcwd()
print("current directory is : " + dirpath)
foldername = os.path.basename(dirpath)
print("Directory name is : " + foldername)

########################

#Código copiado, ajeitar ainda.    
from PIL import Image

def getImagesWithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    cv2.destroyAllWindows() 
    return faces, np.array(IDs)

