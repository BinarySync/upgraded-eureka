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
    
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/lenovo_3.4ghz/null.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='Lenovo_SEM_DETECÇÃO'))

df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/lenovo_3.4ghz/haar_only.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='Lenovo_HaarCascade'))

df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/lenovo_3.4ghz/lbph.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='Lenovo_LBPH'))

df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/lenovo_3.4ghz/fisher.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='Lenovo_FisherFaces'))

df = pd.read_csv(proj_dir+'/tests/desempenho_tests/025_samples_4ids/lenovo_3.4ghz/eigen.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='Lenovo_EigenFaces'))



#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/149_samples_2ids/[]_3.9ghz_video640x360/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x360][]_SEM_DETECÇÃO'))

#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/149_samples_2ids/[]_3.9ghz_video640x360/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x360][]_APENAS_HAARCASCADE'))

    
#fig.show()
fig.update_layout(
        title="Resolução de vídeo: 1280x720 - Resolução dos rostos: 640x640",
        xaxis_title="Número do Frame",
        yaxis_title="Frametime")

from plotly.offline import plot
plot(fig)