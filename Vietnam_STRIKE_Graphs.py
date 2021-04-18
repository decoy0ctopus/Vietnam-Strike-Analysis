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

dataset = pd.read_csv('THOR_Vietnam_Bombing_Operations.csv')

dataset = dataset[dataset['MFUNC_DESC'] == 'STRIKE']


dataset['MSNDATE'] = pd.to_datetime(dataset['MSNDATE'], infer_datetime_format=True, errors='coerce')
dataset['MSNDATE'] = dataset['MSNDATE'].dt.strftime('%Y%m')

 # Get rid of missing data and replace with most frequent

from sklearn.impute import SimpleImputer
        
imputer = SimpleImputer(strategy='most_frequent')
#dataset['WEAPONTYPE'] = dataset['WEAPONTYPE'].fillna("")
dataset['WEAPONTYPE'] = imputer.fit_transform(dataset['WEAPONTYPE'].values.reshape(-1, 1))
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
dataset['WEAPONSLOADEDWEIGHT'] = imputer.fit_transform(dataset['WEAPONSLOADEDWEIGHT'].values.reshape(-1, 1))

dataset = dataset.sort_values(by='MSNDATE')
#import matplotlib.pyplot as plt

# =============================================================================
# plt.scatter(dataset['MSNDATE'], dataset['WEAPONTYPE'])
# plt.xlabel("MSNDATE")
# plt.ylabel("WEAPONTYPE")
# plt.show()
# =============================================================================
