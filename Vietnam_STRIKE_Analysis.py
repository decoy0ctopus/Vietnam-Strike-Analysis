# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:51:43 2020

@author: pater
"""


# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

# Importing the dataset

dataset = pd.read_csv('Vietnam_Bombing_Operations_STRIKE_HIT_YN.csv')

import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster, HeatMapWithTime

# Create a map
m_1 = folium.Map(location=[17.34, 106.71], tiles='Stamen Terrain', zoom_start=8)

# Add points to the map

tgtloc = []
msn_date = []
for idx, row in dataset.iterrows():
    if not math.isnan(row['TGTLONDDD_DDD_WGS84']) and not math.isnan(row['TGTLATDD_DDD_WGS84']):
        tgtloc.append([row['TGTLATDD_DDD_WGS84'], row['TGTLONDDD_DDD_WGS84']])
        msn_date.append(row['MSNDATE'])
# m_1.add_child(mc)

#HeatMap(data=tgtloc, radius=10).add_to(m_1)

HeatMapWithTime(data=tgtloc, index=msn_date, radius=10).add_to(m_1)
# Display the map
m_1.save('index.html')