# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import cv2


img = cv2.imread("D:/image.jpg")
#Usando dois argumentos, o segundo trata de 
#uma chave para decidir usar preto e branco
img = cv2.imread("D:/img.png",0)

resized = cv2.resize(img,   (int(img.shape[1]/2) , int(img.shape[0]/2))   )

print(img)
print(type(img))
print(img.shape)

cv2.imshow("Window",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

rec = cv2.face.EigenFaceRecognizer_create()
rec = cv2.face.FisherFaceRecognizer_create()
rec = cv2.face.LBPHFaceRecognizer_create()

cv2.haarcascades.
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gray_img = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img,scaleFactor=1.05,minNeighbors=5)

cv2.imshow("Window",gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
