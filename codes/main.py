# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
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
#proj_dir = "C:/Users/ALUNO/Documents/NanDS/upgraded-eureka/codes/"
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



#Teste de código para importar imagem a imagem.
def getImageFromPath(imagedir,ID):
    faces = []
    IDs = []
    faceImg = Image.open(imagedir).convert('L')
    faceNp = np.array(faceImg,'uint8')
    
    face = face_cascade.detectMultiScale(
        faceNp,
        scaleFactor=1.3,
        minNeighbors=5
        )
        
    for (x, y, w, h) in face:
        faceNp = cv2.resize(faceNp[y:y+h,x:x+w], (640,640) )
        faces.append(faceNp)
        #os ID das pessoas só podem ser numéricos1
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(1000)
        cv2.destroyAllWindows() 

    #faces.append(faceNp)
    #IDs.append(ID)
    #cv2.imshow("training",faceNp)
    #cv2.waitKey(1000)
    return faces, np.array(IDs)

#rec = cv2.face.EigenFaceRecognizer_create()
#rec = cv2.face.FisherFaceRecognizer_create()
rec = cv2.face.LBPHFaceRecognizer_create()

mode = "photo"
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
    training_faces      ,training_ids       = getImageFromPath(proj_dir+'/midia/'+'nan_1.jpg',1)
    
    training_faces_add  ,training_ids_add   = getImageFromPath(proj_dir+'/midia/'+'nan_2.jpg',1)
    training_faces.extend(training_faces_add)
    training_ids = np.concatenate((training_ids,training_ids_add))
    
    training_faces_add  ,training_ids_add   = getImageFromPath(proj_dir+'/midia/'+'nan_3.jpg',1)
    training_faces.extend(training_faces_add)
    training_ids = np.concatenate((training_ids,training_ids_add))

    training_faces_add  ,training_ids_add   = getImageFromPath(proj_dir+'/midia/'+'nan_4.jpg',1)
    training_faces.extend(training_faces_add)
    training_ids = np.concatenate((training_ids,training_ids_add))    
    
    training_faces_add  ,training_ids_add   = getImageFromPath(proj_dir+'/midia/'+'bin_1.jpg',2)
    training_faces.extend(training_faces_add)
    training_ids = np.concatenate((training_ids,training_ids_add))
    
    training_faces_add  ,training_ids_add   = getImageFromPath(proj_dir+'/midia/'+'bin_2.jpg',2)
    training_faces.extend(training_faces_add)
    training_ids = np.concatenate((training_ids,training_ids_add))
    
    training_faces_add  ,training_ids_add   = getImageFromPath(proj_dir+'/midia/'+'bin_3.jpg',2)
    training_faces.extend(training_faces_add)
    training_ids = np.concatenate((training_ids,training_ids_add))
    
    training_faces_add  ,training_ids_add   = getImageFromPath(proj_dir+'/midia/'+'bin_4.jpg',2)
    training_faces.extend(training_faces_add)
    training_ids = np.concatenate((training_ids,training_ids_add))


#Treinar é uma atividade demorada, e não utiliza vários núcleos, recomenda-se usar vídeos pequenos
rec.train(training_faces,training_ids.astype(int))
rec.save(proj_dir+'trainingData.yml')



#VIDEO SOURCE, SET 0 to Camera
source = 0
#source = proj_dir+'/midia/'+'eri.mp4'

##QUICK BOOT
video_capture = cv2.VideoCapture(source)
if not video_capture.isOpened():
    raise Exception("Erro ao acessar fonte de vídeo")
ret, frame = video_capture.read()
video_capture.release()

##TEXT DEFINITIONS
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

##PRE LOOP SET UP AND CASCADE CLASSIFIER SETTING
video_capture = cv2.VideoCapture(source)
if not video_capture.isOpened():
    raise Exception("Erro ao acessar fonte de vídeo")


##ADDING THE RECOGNIZER AND LOADING TRAINING DATA
#rec = cv2.face.EigenFaceRecognizer_create()
#rec = cv2.face.FisherFaceRecognizer_create()
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read(proj_dir+"trainingData.yml")

##FACE DETECTION LOOP
loops = 0;
length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT));

while (ret and (loops<500 and loops != length-1)):
    ret, frame = video_capture.read()
    #IMG PROCESSING
    ##IMG RESIZING - Lower Res = Higher Speed
    size = 1;
    frame = cv2.resize(frame,   (int(frame.shape[1]*size) , int(frame.shape[0]*size))   )
    
    ##INSERTING TEXT
    cv2.putText(frame,'frame:'+str(loops)+' '+str(length), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

    #Face recognition
    ##Turning Gray (it wasnt used before and worked for detecting faces)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    ##Recognizing
    faces = face_cascade.detectMultiScale(
        frame,
        scaleFactor=1.3,
        minNeighbors=5
        )
    #faces = faces*1/size
    for (x, y, w, h) in faces:
        ids,conf = rec.predict(cv2.resize(frame[y:y+h,x:x+w], (640,640) ))
        #conf=100-float(conf);
        if conf < 50:
            cv2.putText(frame, str(ids)+" "+str(conf), (x+2,y+h-5), font, fontScale, fontColor,lineType)
        else:
            cv2.putText(frame, str(ids)+".No Match: "+str(conf), (x+2,y+h-5), font, fontScale, fontColor,lineType)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Camera Frame",frame)
    #Uma segunda tela aumenta mais ou menos 2% a mais do processamento
    #Mas pode servir para mostrarmos a imagem original e a usada para processar
    #cv2.imshow("Camera Frame2",frame)
    cv2.waitKey(1)
    loops = loops + 1
# Close device
cv2.destroyAllWindows()
video_capture.release()
#FIM TESTE

