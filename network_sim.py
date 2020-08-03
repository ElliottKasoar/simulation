#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:09:11 2020

@author: Elliott
"""

import numpy as np
import pandas as pd
from math import pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from operator import attrgetter
import time
import datetime
from pickle import dump, load
from sklearn.preprocessing import QuantileTransformer


# Standardise data from a pandas dataframe using QuantileTransformer
# QuantileTransformer loaded or calculated (and saved) via load flag
def standardise_data(data, frames, N, load_qt=False):
    
    x = np.concatenate(np.concatenate(data.values, axis=0))
    x = x.reshape(frames, N*6)
    
    if load_qt:
        qt = load(open('transform.pkl', 'rb'))
    else:
        qt = QuantileTransformer(n_quantiles=1000,
                                 output_distribution='normal')
        qt.fit(x)
        dump(qt, open('transform.pkl', 'wb'))
    
    # Standardise data
    x_std = qt.transform(x)
    
    return x_std, qt


# Inverse transform data to recover orginal distribution
def destandardise_data(data_std, qt=None, load_qt=False, orig_data=None):
    
    if (qt is None):
        if load_qt:
            qt = load(open('transform.pkl', 'rb'))
        elif (orig_data is not None):
            qt = QuantileTransformer(n_quantiles=1000,
                                     output_distribution='normal')
            qt.fit(orig_data)
        else:
            print("Unable to destandardise data")
    
    return qt.inverse_transform(data_std)


def save_data(data, frames, N):
    
    x = np.arange(0, N)
    r_lst = ["r" + num for num in x.astype(str)]
    u_lst = ["u" + num for num in x.astype(str)]
    
    columns = [None]*(N*2) 
    columns[::2] = r_lst
    columns[1::2] = u_lst    
    
    sim_df = pd.DataFrame(columns=columns, index=list(range(frames)))
    
    for i in range(frames):

        for j in range(N):
            
            r_name = 'r' + str(j)
            u_name = 'u' + str(j)
            
            idx_1 = 6*j
            idx_2 = 6*j + 3
            idx_3 = 6*j + 6
            
            sim_df.iloc[i][r_name] = data[i][idx_1:idx_2]
            sim_df.iloc[i][u_name] = data[i][idx_2:idx_3]
     
    sim_df.to_pickle('network_values.pkl') 


def main():
    
    # Load saved dataframes
    sim_data = pd.read_pickle('sim_values.pkl') # Data from simulation.py
    const_df = pd.read_pickle('consts.pkl') # Saved constants e.g. N and steps
    
    steps = const_df['steps'][0]
    frames = steps + 1
    N = const_df['N'][0]
    
    data_std, qt = standardise_data(sim_data, frames, N, load_qt=False)
    
    x0 = data_std[0]
    sim_data_0 = sim_data.iloc[0]
   
    data_destd = destandardise_data(data_std, qt)

    save_data(data_destd, frames, N)

    network_data = pd.read_pickle('network_values.pkl') # Data from simulation.py


if __name__ == '__main__':
    main()
