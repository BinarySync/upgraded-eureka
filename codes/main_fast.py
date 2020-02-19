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

#Project Dir
proj_dir = "N:/NeoTokyo_Data/Documents/GitHub/upgraded-eureka/codes/"

face_cascade = cv2.CascadeClassifier(proj_dir+"haarcascade_frontalface_default.xml")
train_res = (640,640)

device = '2AM'

#Existem três REC_MODE, 'LBPH', 'Fisher' e 'Eigen'. Existe um que não usa Reconhecimento, o 'null' e 'haar_only'
#rec_mode = 'Eigen'    

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

    ##VIDEO LOOP START
    while (ret and (loops<500 and loops != length-1)):

        start = time.time()
        ret, frame = video_capture.read()
        
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
            
            #faces = faces*1/size
            for (x, y, w, h) in faces:
                ##Face recognition.Using the Recognizer
                if rec_mode != 'null' and rec_mode != 'haar_only':
                    ids,conf = rec.predict(cv2.resize(frame[y:y+h,x:x+w], train_res ))    
            
        end = time.time()
        seconds = end - start
        ##FrameTime.formating, 6 digit float.
        fps  = "%6f" % seconds
        frametime.append(fps)
        loops = loops + 1
    
    repeat = 1
    while (repeat != 5):
        
        loops = 0
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            raise Exception("Erro ao acessar fonte de vídeo")
        
        while (ret and (loops<500 and loops != length-1)):
            start = time.time()
            ret, frame = video_capture.read()
            
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
                
                #faces = faces*1/size
                for (x, y, w, h) in faces:
                    ##Face recognition.Using the Recognizer
                    if rec_mode != 'null' and rec_mode != 'haar_only':
                        ids,conf = rec.predict(cv2.resize(frame[y:y+h,x:x+w], train_res ))    
                
            end = time.time()
            seconds = end - start
            ##FrameTime.formating, 6 digit float.
            fps  = "%6f" % ((float(frametime[loops+1]) + seconds)/2)
            frametime[loops+1] = fps
    
            loops = loops + 1
        
        repeat = repeat + 1
        
    # Close device
    cv2.destroyAllWindows()
    video_capture.release()
    #FIM TESTE
    
    #Guardando em um CSV
    import pandas as pd 
    pd.DataFrame(frametime).to_csv(proj_dir+"/"+rec_mode+".csv",header=None, index=None)
    return "VideoRes: "+str(frame.shape[1])+"x"+str(frame.shape[0])+" FaceRes: "+str(train_res[0])+"x"+str(train_res[1])

size = 1

null_res = run_recognizer('null',source,size)
run_recognizer('haar_only',source,size)
run_recognizer('LBPH',source,size)
run_recognizer('Eigen',source,size)
run_recognizer('Fisher',source,size)

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
