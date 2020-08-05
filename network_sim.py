#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 21:09:11 2020

@author: Elliott
"""

import numpy as np
import pandas as pd
import time
import datetime
from pickle import dump, load
from sklearn.preprocessing import QuantileTransformer

from keras.optimizers import Adam
from keras.layers import Input, BatchNormalization, LSTM
from keras.layers.core import Dense, Dropout, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model



# Standardise data from a pandas dataframe using QuantileTransformer
# QuantileTransformer loaded or calculated (and saved) via load flag
def standardise_data(data, frames, N, load_qt=False, norm=True):
    
    features = N*6
    
    x = np.concatenate(np.concatenate(data.values, axis=0))
    x = x.reshape(frames, features)
    
    if load_qt:
        qt = load(open('transform.pkl', 'rb'))
    else:
        qt = QuantileTransformer(n_quantiles=1000,
                                 output_distribution='normal')
        qt.fit(x)
        dump(qt, open('transform.pkl', 'wb'))
    
    # Standardise data
    x = qt.transform(x)
    
    if norm:
        scale = x.max(axis=0)
    else:
        scale = np.ones(features)
        
    np.save('scale.npy', scale)
    
    x = np.divide(x, scale)
    
    return x, qt, scale


# Inverse transform data to recover orginal distribution
def destandardise_data(data, qt=None, load_qt=False, orig_data=None,
                       scale=None, norm=True):
    
    if (qt is None):
        if load_qt:
            qt = load(open('transform.pkl', 'rb'))
        elif (orig_data is not None):
            qt = QuantileTransformer(n_quantiles=1000,
                                     output_distribution='normal')
            qt.fit(orig_data)
        else:
            print("Unable to destandardise data")
    
    if (scale is None) and norm:
        scale = np.load('scale.npy')
    
    if norm:
        data = np.multiply(data, scale)
    
    data = qt.inverse_transform(data)
    
    return data


# Sequence data
def seq_data(data, seq_length, input_dim, output_dim):
    
    seq_number = len(data) - seq_length
    
    x = np.zeros([seq_number, seq_length, input_dim])
    y = np.zeros([seq_number, output_dim])
    
    for i in range(seq_number):
        x[i] = data[i:i+seq_length]
        y[i] = data[i+seq_length]
    
    return x, y


# Save data from numpy array as dataframe
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


#Get (Adam) optimiser. Can replace Adam with others if required
#Input: Learning rate, beta_1 and beta_2 for Adam optimiser 
#Returns: Adam optimiser with parameters given
def get_optimizer(lr=0.001, beta_1=0.9, beta_2=0.999):
    
    return Adam(lr=lr, beta_1=beta_1, beta_2=beta_2) 


#Inputs: None 
#Returns: 'binary_crossentropy' loss function currently, 
#Note: can replace with other built in Keras loss functions
def get_loss_function():
    
    return 'mean_squared_error'


def build_network(optimizer, loss_func, seq_length, input_dim, output_dim, input_nodes,
                  internal_nodes, internal_layers):
    
    RNN_input = Input(shape=(seq_length, input_dim), name='RNN_input')
    layer = LSTM(input_nodes, return_sequences=True)(RNN_input) #relu activation?
    layer = Dropout(0.1)(layer)
    layer = LSTM(input_nodes)(layer)
    layer = Dropout(0.1)(layer)
    
      #Internal layers
    for i in range(internal_layers):
        layer = Dense(internal_nodes)(layer)
        layer = LeakyReLU(0.2)(layer)
        layer = Dropout(0.1)(layer)
    
    RNN_output = Dense(output_dim, activation='sigmoid')(layer)
    
    model = Model(inputs=RNN_input, outputs=RNN_output)
    
    model.compile(loss=loss_func, optimizer=optimizer)
    
    return model


def main():
        
    # Load saved dataframes
    sim_data = pd.read_pickle('sim_values.pkl') # Data from simulation.py
    const_df = pd.read_pickle('consts.pkl') # Saved constants e.g. N and steps
    
    steps = const_df['steps'][0]
    frames = steps + 1
    N = const_df['N'][0]

    # Network parameters
    seq_length = 100
    input_dim = N*6
    output_dim = N*6
    input_nodes = 512
    internal_nodes = 256
    internal_layers = 2

    # Training parameters
    batch_size = 128
    epochs = 100
    
    # Adam optimiser parameters 
    lr = 0.0001
    b1 = 0.5
    b2 = 0.9
    
    data, qt, scale = standardise_data(sim_data, frames, N, load_qt=False,
                                       norm=True)
    
    print("Building network...")
    optimizer = get_optimizer(lr, b1, b2)
    loss_func = get_loss_function()
    model = build_network(optimizer, loss_func, seq_length, input_dim, 
                          output_dim, input_nodes, internal_nodes,
                          internal_layers)
    print("Network built")
    
    x, y = seq_data(data, seq_length, input_dim, output_dim)
    
    # x0 = data[0]
    # sim_data_0 = sim_data.iloc[0]
   
    print("Saving data...")
    data = destandardise_data(data, qt, scale, norm=True)
    save_data(data, frames, N)
    print("Data saved")
    
    # network_data = pd.read_pickle('network_values.pkl')


if __name__ == '__main__':
    main()
