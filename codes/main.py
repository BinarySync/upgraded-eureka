# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

########################
########################
########################
########################
########################

#PRE_SCRIPT_ARGUMENTS
import cv2
import numpy as np

#Project Dir
proj_dir = "N:/NeoTokyo_Data/Documents/GitHub/upgraded-eureka/codes/"
#proj_dir = "D:/Git/upgraded-eureka/codes/"
#proj_dir = "C:/Users/Guilherme/Desktop/TCC/upgraded-eureka/codes/"
#proj_dir = "C:/Users/ALUNO/Documents/GitHub/upgraded-eureka/codes/"

face_cascade = cv2.CascadeClassifier(proj_dir+"haarcascade_frontalface_default.xml")
train_res = (640,640)


device = '2AM'

#Existem três REC_MODE, 'LBPH', 'Fisher' e 'Eigen'. Existe um que não usa Reconhecimento, o 'null' e 'haar_only'
#rec_mode = 'Eigen'    

#Existem 2 tipos de treinamento(gerar os arrays para treinamento), o Photo e video
train_mode = "photo"
#train_mode = "null"

#############################
#############################
#############################

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
                #os ID das pessoas só podem ser numéricos
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
            faceNp = cv2.resize(faceNp[y:y+h,x:x+w], train_res )
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

if train_mode=="video":
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
if train_mode == "photo":
    
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

def run_trainer(rec_mode,training_faces,training_ids):
    if rec_mode == 'Eigen':
        rec = cv2.face.EigenFaceRecognizer_create()
    
    if rec_mode == 'Fisher':
        rec = cv2.face.FisherFaceRecognizer_create()
    
    if rec_mode == 'LBPH':
        rec = cv2.face.LBPHFaceRecognizer_create()
    
    if rec_mode != 'null' and rec_mode != 'haar_only':
        rec.train(training_faces,training_ids.astype(int))
        rec.save(proj_dir+rec_mode+'_trainingData.yml')

run_trainer('LBPH',training_faces,training_ids)
run_trainer('Eigen',training_faces,training_ids)
run_trainer('Fisher',training_faces,training_ids)

#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################

#VIDEO SOURCE, SET 0 to Camera
#source = 0
source = proj_dir+'/midia/'+'video_bin.mp4'

def run_recognizer(rec_mode, source, frame_size):
    #Frame_size é um multiplicador, 1 para a resolução atual e 0.5 para metade, etc etc.
    
    ##QUICK BOOT(AVOIDS PROBLEMS INTO LOADING)
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
    
    ##PRE LOOP SET UP
    video_capture = cv2.VideoCapture(source)
    if not video_capture.isOpened():
        raise Exception("Erro ao acessar fonte de vídeo")
    
    ##ADDING THE RECOGNIZER AND LOADING TRAINING DATA
    if rec_mode == 'Eigen':
        rec = cv2.face.EigenFaceRecognizer_create()
    
    if rec_mode == 'Fisher':
        rec = cv2.face.FisherFaceRecognizer_create()
    
    if rec_mode == 'LBPH':
        rec = cv2.face.LBPHFaceRecognizer_create()   
    
    if rec_mode != 'null' and rec_mode != 'haar_only':
        rec.read(proj_dir+rec_mode+"_trainingData.yml")
    
    ##VIDEO LOOP SET UP
    loops = 0;
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT));
    
    ##FRAMETIME SET UP
    import time
    frametime = ['frametime']
    
    #last_conf = -1

    ##VIDEO LOOP START
    while (ret and (loops<500 and loops != length-1)):
        #FrameTime
        ##FrameTime.starting time
        start = time.time()
        
        #IMG PROCESSING
        ret, frame = video_capture.read()
        
        ##IMG PROCESSING.RESIZING - Lower Res = Higher Speed.
        #frame_size = 1;
        frame = cv2.resize(frame,   (int(frame.shape[1]*frame_size) , int(frame.shape[0]*frame_size))   )
    
        ##IMG PROCESSING.Turning Image Gray (it wasnt used before and worked for detecting faces, but is needed for Recognizer)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        #Face recognition  
        ##Face recognition.HaarCascate, Detecting faces
        if rec_mode != 'null':
            faces = face_cascade.detectMultiScale(
                frame,
                scaleFactor=1.3,
                minNeighbors=5
                )
            ids = '--'
            conf = 10000000
        
            #faces = faces*1/size
            for (x, y, w, h) in faces:
                ##Face recognition.Using the Recognizer
                if rec_mode != 'null' and rec_mode != 'haar_only':
                    ids,conf = rec.predict(cv2.resize(frame[y:y+h,x:x+w], train_res ))    
                if conf < 30:
                    cv2.putText(frame, str(ids)+". "+str(conf), (x+2,y+h-5), font, fontScale, fontColor,lineType)
                else:
                    cv2.putText(frame, str(ids)+".No Match: "+str(conf), (x+2,y+h-5), font, fontScale, fontColor,lineType)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        end = time.time()
        seconds = end - start
        ##FrameTime.formating, 6 digit float.
        fps  = "%6f" % seconds
        #if last_conf == conf:
        #    conf = '--'
        #last_conf = conf
        #frametime.extend([[fps,conf]])
        ##FrameTime.Add to graph
        frametime.extend([fps])
    
        ##IMG PROCESSING.INSERTING TEXT on TOPLEFT
        cv2.putText(frame,'FPS: '+str(fps)+' Frame:'+str(loops)+' '+str(length), 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            lineType)
        
        cv2.imshow("Camera Frame ",frame)
        #Uma segunda tela aumenta mais ou menos 2% a mais do processamento
        #Mas pode servir para mostrarmos a imagem original e a usada para processar
        #cv2.imshow("Camera Frame2",frame)
        cv2.waitKey(1)
        loops = loops + 1
    # Close device
    cv2.destroyAllWindows()
    video_capture.release()
    #FIM TESTE
    
    #Guardando em um CSV
    import pandas as pd 
    pd.DataFrame(frametime).to_csv(proj_dir+"/"+rec_mode+".csv",header=None, index=None)
    return "VideoRes: "+str(frame.shape[1])+"x"+str(frame.shape[0])+" FaceRes: "+str(train_res[0])+"x"+str(train_res[1])

null_res = run_recognizer('null',source,1)
run_recognizer('haar_only',source,1)
run_recognizer('LBPH',source,1)
run_recognizer('Eigen',source,1)
run_recognizer('Fisher',source,1)

#Importando de CSV
import pandas as pd
graph_mode = 'go'

if graph_mode == 'px':
    import plotly.express as px
    
    df = pd.read_csv(proj_dir+'/file.csv')
    fig = px.line(df, x=df.index, y='frametime' ,title='Frametimes')
    
    df = pd.read_csv(proj_dir+'/file_1.csv')
    fig.add_scatter(x=df.index,y=df['frametime'], mode='lines')

if graph_mode == 'go':
    import plotly.graph_objects as go
    fig = go.Figure()
    
    df = pd.read_csv(proj_dir+'/null.csv')
    fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name=device+'_NO_RECOGNITION'))
    
    df = pd.read_csv(proj_dir+'/haar_only.csv')
    fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name=device+'_ONLY_HAARCASCATE'))
    
    df = pd.read_csv(proj_dir+'/LBPH.csv')
    fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name=device+'_LBPH'))
    
    df = pd.read_csv(proj_dir+'/Eigen.csv')
    fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name=device+'_EigenFaces'))
    
    df = pd.read_csv(proj_dir+'/Fisher.csv')
    fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name=device+'_FisherFaces'))
    #fig.show()
    fig.update_layout(
            title="Frametimes at "+null_res,
            xaxis_title="Framenumber",
            yaxis_title="Frametime")

from plotly.offline import plot
plot(fig)
