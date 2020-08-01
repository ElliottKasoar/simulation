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

sim_data = pd.read_pickle('sim_values.pkl')
init_values = pd.read_pickle('init_values.pkl')
r = sim_data.iloc[0]['r0'][0]
data = r

x = np.concatenate(np.concatenate(sim_data.values, axis=0)).reshape(6000,36)
x0 = x[0]
sim_data_0 = sim_data.iloc[0]

transform = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
transform.fit(x)
x_norm = transform.transform(x)


# test = np.zeros([10000, 1])
# test[:,0] = np.linspace(-1000, 1000, 10000)

# # qt = load(open('transform.pkl', 'rb'))
# transform = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
# transform.fit(test)

# test_norm = transform.transform(test)

# test_unnorm = transform.inverse_transform(test_norm)

# test2 = test / 2
# test2_norm = transform.transform(test2)

# plt.hist(test_norm, bins=100)
# plt.hist(test2_norm, bins=int(100))

# dump(transform, open('transform.pkl', 'wb'))