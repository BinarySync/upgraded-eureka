# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:00:39 2020

@author: fewii
"""
proj_dir = "N:/NeoTokyo_Data/Documents/GitHub/upgraded-eureka/codes/"
#proj_dir = "C:/Users/ALUNO/Documents/GitHub/upgraded-eureka/codes/"

#Importando de CSV
import pandas as pd

import plotly.graph_objects as go
fig = go.Figure()

#Camera
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/camera/test_5_difres/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='2AM_SEM_DETECÇÃO_COM_CAMERA'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/camera/test_5_difres/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='2AM_APENAS_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/camera/test_7_difres/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='2AM_LBPH'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/camera/test_7_difres/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='2AM_FisherFaces'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/camera/test_7_difres/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='2AM_EigenFaces'))

#Dispositivo
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/lenovofix_3.4ghz/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='Lenovo_SEM_DETECÇÃO'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/lenovoold_3.4ghz/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='Lenovo_HaarCascade'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/lenovofix_3.4ghz/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='Lenovo_LBPH'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/lenovofix_3.4ghz/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='Lenovo_FisherFaces'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/lenovofix_3.4ghz/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='Lenovo_EigenFaces'))

#Resolução
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_resolution/2am_3.9ghz[640x360]/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x360]2AM_SEM_DETECÇÃO'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_resolution/2am_3.9ghz[640x360]/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x360]2AM_APENAS_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_resolution/2am_3.9ghz[960x540]/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[960x540]2AM_SEM_DETECÇÃO'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_resolution/2am_3.9ghz[960x540]/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[960x540]2AM_APENAS_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.9ghz/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1280x720]2AM_SEM_DETECÇÃO'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.9ghz/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1280x720]2AM_APENAS_HAARCASCADE'))

#FaceRes
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_faceres/2am_3.9ghz[160x160]/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='2AM_SEM_DETECÇÃO'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_faceres/2am_3.9ghz[160x160]/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='2AM_HaarCascade'))

#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_faceres/2am_3.9ghz[160x160]/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[160x160]2AM_FisherFaces'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_faceres/2am_3.9ghz[320x320]/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[320x320]2AM_FisherFaces'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_faceres/2am_3.9ghz[480x480]/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[480x480]2AM_FisherFaces'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.9ghz/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x640]2AM_FisherFaces'))

#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_faceres/2am_3.9ghz[160x160]/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[160x160]2AM_EigenFaces'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_faceres/2am_3.9ghz[320x320]/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[320x320]2AM_EigenFaces'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_faceres/2am_3.9ghz[480x480]/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[480x480]2AM_EigenFaces'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.9ghz/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x640]2AM_EigenFaces'))

#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_faceres/2am_3.9ghz[160x160]/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[160x160]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_faceres/2am_3.9ghz[320x320]/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[320x320]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_faceres/2am_3.9ghz[480x480]/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[480x480]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.9ghz/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x640]2AM_LBPH'))


##Hyperthread
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_1C/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1 Núcleo]2AM_SEM_DETECÇÃO'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_1CHT[2]/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1 Núcleo e HyperThread]2AM_SEM_DETECÇÃO'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_2C[2]/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2 Núcleos]2AM_SEM_DETECÇÃO'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_2CHT[1]/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2 Núcleos e HyperThread]2AM_SEM_DETECÇÃO'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_3C[2]/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3 Núcleos]2AM_SEM_DETECÇÃO'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_3CHT[2]/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3 Núcleos e HyperThread]2AM_SEM_DETECÇÃO'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_4C/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4 Núcleos]2AM_SEM_DETECÇÃO'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_4CHT/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4 Núcleos e HyperThread]2AM_SEM_DETECÇÃO'))

#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_1C/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1 Núcleo]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_1CHT[1]/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1 Núcleo e HyperThread]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_2C[2]/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2 Núcleos]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_2CHT[1]/haar_only.csv')
##fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2 Núcleos e HyperThread]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_3C[2]/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3 Núcleos]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_3CHT[2]/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3 Núcleos e HyperThread]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_4C/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4 Núcleos]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_4CHT/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4 Núcleos e HyperThread]2AM_HAARCASCADE'))

#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_1C/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1 Núcleo]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_1CHT[1]/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1 Núcleo e HyperThread]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_2C[2]/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2 Núcleos]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_2CHT[1]/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2 Núcleos e HyperThread]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_3C[2]/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3 Núcleos]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_3CHT[2]/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3 Núcleos e HyperThread]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_4C/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4 Núcleos]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_4CHT/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4 Núcleos e HyperThread]2AM_Fisher'))

#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_1C/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1 Núcleo]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_1CHT[1]/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1 Núcleo e HyperThread]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_2C[2]/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2 Núcleos]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_2CHT[1]/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2 Núcleos e HyperThread]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_3C[2]/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3 Núcleos]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_3CHT[2]/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3 Núcleos e HyperThread]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_4C/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4 Núcleos]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_4CHT/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4 Núcleos e HyperThread]2AM_Eigen'))

#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_1C/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1 Núcleo]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_1CHT[1]/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1 Núcleo e HyperThread]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_2C[2]/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2 Núcleos]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_2CHT[1]/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2 Núcleos e HyperThread]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_3C[2]/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3 Núcleos]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_3CHT[2]/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3 Núcleos e HyperThread]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_4C/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4 Núcleos]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_core/2AM_3.9GHZ_4CHT/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4 Núcleos e HyperThread]2AM_LBPH'))

#Tempo por amostra
#graph = {'index':[25,149,274,277,326,402], 'frametime':[0.0564,0.0611,0.0661,0.0672,0.0679,0.0723]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.frametime , name='Amostras_LBHP'))
#graph = {'index':[25,149,274,277,326,402], 'frametime':[0.0301,0.1103,0.2066,0.2125,0.2435,0.3358]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.frametime , name='Amostras_Eigen'))
#graph = {'index':[25,149,274,277,326,402], 'frametime':[0.0238,0.0226,0.0239,0.0238,0.0248,0.0236]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.frametime , name='Amostras_Fisher'))

#Nucleos-Haar
#graph = {'index':[1,1.5,2,2.5,3,3.5,4,4.5], 'clock':[0.0672,0.0573,0.0352,0.0318,0.0264,0.0244,0.0218,0.0205]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='Haar_Desempenho_por_núcleo'))
#graph = {'index':[1,1.5,2,2.5,3,3.5,4,4.5], 'clock':[0.0699,0.0603,0.0382,0.0345,0.0291,0.0272,0.0245,0.0232]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='Fisher_Desempenho_por_núcleo'))
#graph = {'index':[1,1.5,2,2.5,3,3.5,4,4.5], 'clock':[0.0765,0.0673,0.0448,0.0409,0.0354,0.0335,0.0308,0.0296]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='Eigen_Desempenho_por_núcleo'))
#graph = {'index':[1,1.5,2,2.5,3,3.5,4,4.5], 'clock':[0.1027,0.0948,0.0711,0.0679,0.0623,0.0601,0.0581,0.0565]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='LBPH_Desempenho_por_núcleo'))

##Nucleos-Haar
#graph = {'index':[1,1.5,2,2.5,3,3.5,4,4.5], 'clock':[0.0699-(0.0672),0.0603-(0.0573),0.0382-(0.0352),0.0345-(0.0318),0.0291-(0.0264),0.0272-(0.0244),0.0245-(0.0218),0.0232-(0.0205)]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='Fisher_Desempenho_por_núcleo'))
#graph = {'index':[1,1.5,2,2.5,3,3.5,4,4.5], 'clock':[0.0765-(0.0672),0.0673-(0.0573),0.0448-(0.0352),0.0409-(0.0318),0.0354-(0.0264),0.0335-(0.0244),0.0308-(0.0218),0.0296-(0.0205)]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='Eigen_Desempenho_por_núcleo'))
#graph = {'index':[1,1.5,2,2.5,3,3.5,4,4.5], 'clock':[0.1027-(0.0672),0.0948-(0.0573),0.0711-(0.0352),0.0679-(0.0318),0.0623-(0.0264),0.0601-(0.0244),0.0581-(0.0218),0.0565-(0.0205)]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='LBPH_Desempenho_por_núcleo'))

##Testes Clock

#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.3ghz/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.3GHz]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.5ghz/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.5GHz]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.7ghz/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.7GHz]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.9ghz/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.9GHz]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.1ghz/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.1GHz]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.3ghz/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.3GHz]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.5ghz/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.5GHz]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.7ghz/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.7GHz]2AM_HAARCASCADE'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.9ghz/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.9GHz]2AM_HAARCASCADE'))

#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.3ghz/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.3GHz]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.5ghz/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.5GHz]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.7ghz/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.7GHz]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.9ghz/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.9GHz]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.1ghz/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.1GHz]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.3ghz/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.3GHz]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.5ghz/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.5GHz]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.7ghz/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.7GHz]2AM_Fisher'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.9ghz/fisher.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.9GHz]2AM_Fisher'))

#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.3ghz/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.3GHz]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.5ghz/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.5GHz]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.7ghz/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.7GHz]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.9ghz/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.9GHz]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.1ghz/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.1GHz]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.3ghz/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.3GHz]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.5ghz/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.5GHz]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.7ghz/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.7GHz]2AM_Eigen'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.9ghz/eigen.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.9GHz]2AM_Eigen'))

#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.3ghz/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.3GHz]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.5ghz/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.5GHz]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.7ghz/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.7GHz]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_2.9ghz/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2.9GHz]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.1ghz/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.1GHz]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.3ghz/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.3GHz]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.5ghz/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.5GHz]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.7ghz/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.7GHz]2AM_LBPH'))
#df = pd.read_csv(proj_dir+'tests/desempenho_tests/025_samples_4ids/tests_clock/2am_3.9ghz/lbph.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3.9GHz]2AM_LBPH'))

#Clock
#graph = {'index':[2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9],'clock':[0.0349,0.0325,0.0295,0.0273,0.0264,0.0240,0.0227,0.0215,0.0205]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='Haar_Desempenho_por_clock'))
#graph = {'index':[2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9],'clock':[0.0387,0.0358,0.0329,0.0306,0.0287,0.0268,0.0253,0.0241,0.0238]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='Fisher_Desempenho_por_clock'))
#graph = {'index':[2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9],'clock':[0.0470,0.0436,0.0400,0.0375,0.0355,0.0333,0.0319,0.0301,0.0301]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='Eigen_Desempenho_por_clock'))
#graph = {'index':[2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9],'clock':[0.0954,0.0876,0.0813,0.0752,0.0709,0.0661,0.0623,0.0591,0.0564]}
#graph = pd.DataFrame(data=graph)
#fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='LBPH_Desempenho_por_clock'))


graph = {'index':[2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9],
         'clock':[0.0387-(0.0349),0.0358-(0.0325),0.0329-(0.0295),0.0306-(0.0273),0.0287-(0.0264),0.0268-(0.0240),0.0253-(0.0227),0.0241-(0.0215),0.0238-(0.0205)]}
graph = pd.DataFrame(data=graph)
fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='Fisher_Desempenho_por_clock'))
graph = {'index':[2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9],
         'clock':[0.0470-(0.0349),0.0436-(0.0325),0.0400-(0.0295),0.0375-(0.0273),0.0355-(0.0264),0.0333-(0.0240),0.0319-(0.0227),0.0301-(0.0215),0.0301-(0.0205)]}
graph = pd.DataFrame(data=graph)
fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='Eigen_Desempenho_por_clock'))
graph = {'index':[2.3,2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9],
         'clock':[0.0954-(0.0349),0.0876-(0.0325),0.0813-(0.0295),0.0752-(0.0273),0.0709-(0.0264),0.0661-(0.0240),0.0623-(0.0227),0.0591-(0.0215),0.0564-(0.0205)]}
graph = pd.DataFrame(data=graph)
fig.add_trace(go.Scatter(x=graph['index'], y=graph.clock , name='LBPH_Desempenho_por_clock'))

#fig.show()
fig.update_layout(
        title="Resolução do vídeo: 1280x720 - Resolução dos rostos: 640x640",
        xaxis_title="Clock do processador",
        yaxis_title="Frametime (s)")

from plotly.offline import plot
plot(fig)