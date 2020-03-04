# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:56:38 2020

@author: ALUNO
"""


########################
########################
########################
########################
########################

#TESTING
import cv2
import numpy as np

#Project Dir
#import os
#proj_dir = os.getcwd()
#proj_dir = proj_dir + "\\"
proj_dir = "N:/NeoTokyo_Data/Documents/GitHub/upgraded-eureka/codes/"
#proj_dir = "D:/Git/upgraded-eureka/codes/"
#proj_dir = "C:/Users/Guilherme/Desktop/TCC/upgraded-eureka/codes/"
#proj_dir = "C:/Users/ALUNO/Documents/GitHub/upgraded-eureka/codes/"
face_cascade = cv2.CascadeClassifier(proj_dir+"haarcascade_frontalface_default.xml")

#ESSE METODO TREINA APENAS PARA UMA PESSOA, PARA VÁRIAS
#TEMOS DE ARRANJAR UM JEITO DE CARREGAR MAIS VÍDEOS E DIZER QUAL É O ID DE CADA VIDEO
#O RECONHECIMENTO FUNCIONA DIZENDO NO TRAINING QUEM É O QUE, PELAS LABEL E FRAMES
#Aparentemente, treinar só uma pessoa faz o código não funcionar, ele não detecta outras pessoas.
def getImagesFromVideo(source,id):
    faces = []
    ids = []
    x_old,y_old,w_old,h_old = [0,0,0,0]
    video_capture = cv2.VideoCapture(source)
    if not video_capture.isOpened():
        raise Exception("Erro ao acessar fonte de vídeo")
    ret, frame = video_capture.read()
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    while(ret and len(faces) != length-1):
        #As imagens tem de estar em preto em branco para o LBPH aceitar.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face = face_cascade.detectMultiScale(
        frame,
        scaleFactor=1.3,
        minNeighbors=5
        )
        
        for (x, y, w, h) in face:
            if x != x_old and y != y_old and w != w_old and h != h_old:
                x_old,y_old,w_old,h_old = [x,y,w,h]
                faces.append(frame[y:y+h,x:x+w])
                #os ID das pessoas só podem ser numéricos1
                ids.append(id)
                cv2.imshow("training",frame[y:y+h,x:x+w])
                cv2.waitKey(10)
        ret, frame = video_capture.read()
        
    cv2.destroyAllWindows() 
    return faces,np.array(ids)

    
#Teste de código para importar imagem a imagem
from PIL import Image

def getImageFromPath(imagedir, elements):
    faces = []
    IDs = []
    for element in elements:    
        print(element)
        faceImg = Image.open(imagedir+element[0]).convert('L')
        faceNp = np.array(faceImg,'uint8')
        
        face = face_cascade.detectMultiScale(
            faceNp,
            scaleFactor=1.3,
            minNeighbors=5
            )
            
        for (x, y, w, h) in face:
            print(face)
            faceNp = cv2.resize(faceNp[y:y+h,x:x+w], (640,640) )
            faces.append(faceNp)
            #os ID das pessoas só podem ser numéricos1
            IDs.append(element[1])
            cv2.imshow("training",faceNp)
            cv2.waitKey(250)
            cv2.destroyAllWindows() 

    #faces.append(faceNp)
    #IDs.append(ID)
    #cv2.imshow("training",faceNp)
    #cv2.waitKey(1000)
    return faces, np.array(IDs)

def train(mode):
    if mode=="video":
    #Adding to the training array with VIDEO
        training_faces      ,training_ids       = getImagesFromVideo(proj_dir+'video1.mp4',1)
        training_faces_add  ,training_ids_add   = getImagesFromVideo(proj_dir+'video3.mp4',3)
        
        training_faces      ,training_ids       = getImagesFromVideo(proj_dir+'/midia/'+'nan.mp4',1)
        
        training_faces_add  ,training_ids_add   = getImagesFromVideo(proj_dir+'/midia/'+'bin.mp4',2)
        training_faces.extend(training_faces_add)
        training_ids = np.concatenate((training_ids,training_ids_add))
        
        training_faces_add  ,training_ids_add   = getImagesFromVideo(proj_dir+'/midia/'+'mar.mp4',3)
        training_faces.extend(training_faces_add)
        training_ids = np.concatenate((training_ids,training_ids_add))
        
        training_faces_add  ,training_ids_add   = getImagesFromVideo(proj_dir+'/midia/'+'car.mp4',4)
        training_faces.extend(training_faces_add)
        training_ids = np.concatenate((training_ids,training_ids_add))
        
        training_faces_add  ,training_ids_add   = getImagesFromVideo(proj_dir+'/midia/'+'eri.mp4',5)
        training_faces.extend(training_faces_add)
        training_ids = np.concatenate((training_ids,training_ids_add))
    
    
    ##extend junta os arrays
    ##training_faces.extend(training_faces_add)
    ##os elementos dentro do concatenate tem de ser uma lista de arrays, por isso o "( )" entre os elementos
    ##training_ids = np.concatenate((training_ids,training_ids_add))
    ##ids.extend(ids_add)
    
    ##extend junta os arrays
    #Adding to the training array with IMAGE
    if mode == "photo":
        
        training_faces      ,training_ids       = getImageFromPath(proj_dir+'/midia/',[['nan_1.jpg',1],
                                                                                       ['nan_2.jpg',1],
                                                                                       ['nan_3.jpg',1],
                                                                                       ['nan_4.jpg',1],
                                                                                       ['nan_5.jpg',1],
                                                                                       ['nan_6.jpg',1],
                                                                                       #['nan_7.jpg',1],
                                                                                       #['nan_8.jpg',1],
                                                                                       #['nan_9.jpg',1],
                                                                                       ['bin_1.jpg',2],
                                                                                       ['bin_2.jpg',2],
                                                                                       ['bin_3.jpg',2],
                                                                                       ['bin_4.jpg',2],
                                                                                       ['bin_5.jpg',2],
                                                                                       ['bin_6.jpg',2],
                                                                                       ['bin_7.jpg',2],
                                                                                       ['eri_1.jpg',3],                                                                                 
                                                                                       ['eri_2.jpg',3],
                                                                                       ['eri_3.jpg',3],
                                                                                       ['eri_4.jpg',3],
                                                                                       ['eri_5.jpg',3],
                                                                                       ['eri_6.jpg',3],
                                                                                       ['eri_7.jpg',3],
                                                                                       #['gab_1.jpg',3],
                                                                                       #['gab_2.jpg',3],
                                                                                       #['gab_3.jpg',3],
                                                                                       #['gab_4.jpg',3],
                                                                                       #['gab_5.jpg',3],
                                                                                       #['gab_6.jpg',3],
                                                                                       ['lin_1.jpg',4],
                                                                                       ['lin_2.jpg',4],
                                                                                       ['lin_3.jpg',4],
                                                                                       ['lin_4.jpg',4],
                                                                                       ['lin_5.jpg',4],
                                                                                                      ])
    
    
    
    #Treinar é uma atividade demorada, e não utiliza vários núcleos, recomenda-se usar vídeos pequenos
    
    rec = cv2.face.EigenFaceRecognizer_create()
    #rec = cv2.face.FisherFaceRecognizer_create()
    #rec = cv2.face.LBPHFaceRecognizer_create()
    
    rec.train(training_faces,training_ids.astype(int))
    rec.save(proj_dir+'trainingData.yml')


#######################################
#######################################
#######################################
#######################################