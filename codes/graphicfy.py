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
    
df = pd.read_csv(proj_dir+'/tests/desempenho_tests/149_samples_2ids/2am_3.9ghz/null.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1280x720]2AM_SEM_DETECÇÃOr'))

df = pd.read_csv(proj_dir+'/tests/desempenho_tests/149_samples_2ids/2am_3.9ghz/haar_only.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[1280x720]2AM_APENAS_HAARCASCADE'))

df = pd.read_csv(proj_dir+'/tests/desempenho_tests/149_samples_2ids/2am_3.9ghz_video640x360/null.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x360]2AM_SEM_DETECÇÃO'))

df = pd.read_csv(proj_dir+'/tests/desempenho_tests/149_samples_2ids/2am_3.9ghz_video640x360/haar_only.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x360]2AM_APENAS_HAARCASCADE'))

#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/149_samples_2ids/[]_3.9ghz_video640x360/null.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x360][]_SEM_DETECÇÃO'))

#df = pd.read_csv(proj_dir+'/tests/desempenho_tests/149_samples_2ids/[]_3.9ghz_video640x360/haar_only.csv')
#fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[640x360][]_APENAS_HAARCASCADE'))

    
#fig.show()
fig.update_layout(
        title="Comparação de Frametimes (1280x720 e 640x360)",
        xaxis_title="Número do Frame",
        yaxis_title="Frametime")

from plotly.offline import plot
plot(fig)