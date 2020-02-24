# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:00:39 2020

@author: fewii
"""
proj_dir = "N:/NeoTokyo_Data/Documents/GitHub/upgraded-eureka/codes/"

#Importando de CSV
import pandas as pd

import plotly.graph_objects as go
fig = go.Figure()
    
df = pd.read_csv(proj_dir+'/tests/025_samples_4ids/2am_3.9ghz/null.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='2AM_NO_RECOGNITION'))

df = pd.read_csv(proj_dir+'/tests/025_samples_4ids/2am_3.9ghz/haar_only.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='2AM_Haar_Only'))
 
df = pd.read_csv(proj_dir+'/tests/025_samples_4ids/2am_3.9ghz/LBPH.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[25]2AM_LBPH'))

df = pd.read_csv(proj_dir+'/tests/149_samples_2ids/2am_3.9ghz/LBPH.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[149]2AM_LBPH'))
    
df = pd.read_csv(proj_dir+'/tests/274_samples_3ids/2am_3.9ghz/LBPH.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[274]2AM_LBPH'))

df = pd.read_csv(proj_dir+'/tests/277_samples_4ids/2am_3.9ghz/LBPH.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[277]2AM_LBPH'))

df = pd.read_csv(proj_dir+'/tests/326_samples_4ids/2am_3.9ghz/LBPH.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[324]2AM_LBPH'))

df = pd.read_csv(proj_dir+'/tests/402_samples_5ids/2am_3.9ghz/LBPH.csv')
fig.add_trace(go.Scatter(x=df.index, y=df['frametime'] , name='[402]2AM_LBPH'))
    
#fig.show()
fig.update_layout(
        title="Frametimes comparison (Number of Samples - LBPH)",
        xaxis_title="Framenumber",
        yaxis_title="Frametime")

from plotly.offline import plot
plot(fig)