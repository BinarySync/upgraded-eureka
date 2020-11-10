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
    
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_1core_3.9ghz/null.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1Core - 3.9GHz]2AM_SEM_DETECÇÃO'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_1core_3.9ghz/haar_only.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1Core - 3.9GHz]2AM_HaarCascade'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_1core_3.9ghz/lbph.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1Core - 3.9GHz]2AM_LBPH'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_1core_3.9ghz/fisher.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1Core - 3.9GHz]2AM_FisherFaces'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_1core_3.9ghz/eigen.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1Core - 3.9GHz]2AM_EigenFaces'))

df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_2core_3.9ghz/null.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2Cores - 3.9GHz]2AM_SEM_DETECÇÃO'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_2core_3.9ghz/haar_only.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2Cores - 3.9GHz]2AM_HaarCascade'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_2core_3.9ghz/lbph.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2Cores - 3.9GHz]2AM_LBPH'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_2core_3.9ghz/fisher.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2Cores - 3.9GHz]2AM_FisherFaces'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_2core_3.9ghz/eigen.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[2Cores - 3.9GHz]2AM_EigenFaces'))

df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_3core_3.9ghz/null.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3Cores - 3.9GHz]2AM_SEM_DETECÇÃO'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_3core_3.9ghz/haar_only.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3Cores - 3.9GHz]2AM_HaarCascade'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_3core_3.9ghz/lbph.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3Cores - 3.9GHz]2AM_LBPH'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_3core_3.9ghz/fisher.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3Cores - 3.9GHz]2AM_FisherFaces'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_3core_3.9ghz/eigen.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[3Cores - 3.9GHz]2AM_EigenFaces'))

df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_3.9ghz/null.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4Cores - 3.9GHz]2AM_SEM_DETECÇÃO'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_3.9ghz/haar_only.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4Cores - 3.9GHz]2AM_HaarCascade'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_3.9ghz/lbph.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4Cores - 3.9GHz]2AM_LBPH'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_3.9ghz/fisher.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4Cores - 3.9GHz]2AM_FisherFaces'))
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/2am_3.9ghz/eigen.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[4Cores - 3.9GHz]2AM_EigenFaces'))



#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/149_samples_2ids/[]_3.9ghz_video640x360/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x360][]_SEM_DETECÇÃO'))

#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/149_samples_2ids/[]_3.9ghz_video640x360/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x360][]_APENAS_HAARCASCADE'))

    
#fig.show()
fig.update_layout(
        title="Comparação de Frametimes (1280x720) utilizando 1 a 4 Cores",
        xaxis_title="Número do Frame",
        yaxis_title="Frametime")

from plotly.offline import plot
plot(fig)