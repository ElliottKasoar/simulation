#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:02:11 2020

@author: Elliott
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import time
import datetime


#Plot single frame
def plot_box(init_values, data, box_shape, N):
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, box_shape[0])
    ax.set_ylim(0, box_shape[1])
    ax.set_aspect('equal')
    
    ax.tick_params(
    axis='both',
    which='both',
    bottom=False,
    left=False,
    labelbottom=False,
    labelleft=False)
    
    circles= []
    
    for i in range(N):
        
        colour = init_values.iloc[i]['colour']
        rad = init_values.iloc[i]['rad']
        pos_str = 'r' + str(i)
        r = data.iloc[0][pos_str][0]
        
        circle = Circle(xy=r, radius=rad, color=colour)
        ax.add_patch(circle)        
        circles.append(circle)

    plt.show(fig)
    fig.savefig('Box.pdf', dps=1000)
    plt.close(fig)


def draw_circle(ax, r, rad, colour):
    
    circle = Circle(xy=r, radius=rad, color=colour)
    ax.add_patch(circle)
    
    return circle
    

def update_anim(frame, init_values, data, N, ax, t1, verbose, update_freq,
                frames):
   
    ax.clear()

    circles = []

    for i in range(N):
        
        colour = init_values.iloc[i]['colour']
        rad = init_values.iloc[i]['rad']
        
        pos_str = 'r' + str(i)
        
        r = data.iloc[frame][pos_str]
        
        circle = Circle(xy=r, radius=rad, color=colour)
        ax.add_patch(circle)        
        circles.append(circle)

        # circles.append(draw_circle(ax, r, rad, colour))
        
        
    if verbose:
        frame_update = np.linspace(1, update_freq-1, update_freq-1)
        frame_update *= frames//update_freq
        
        if frame in frame_update:
            t2 = time.time()
            total_t = t2 - t1
            long_time = str(datetime.timedelta(seconds=total_t))
            print(f'Animation progress: {(100 * frame/frames):.1f}%. '
                  f'Current time taken for simulation: {long_time}')
        
    return circles


def animate(init_values, data, box_shape, frames, N, t1, verbose, update_freq):
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, box_shape[0])
    ax.set_ylim(0, box_shape[1])
    ax.set_aspect('equal')

    ax.tick_params(
    axis='both',
    which='both',
    bottom=False,
    left=False,
    labelbottom=False,
    labelleft=False)
    
    anim = FuncAnimation(fig, update_anim, frames=frames, interval=2,
                         blit=True, fargs=(init_values, data, N, ax, t1,
                                           verbose, update_freq, frames))
    
    anim.save('particles_read.gif', writer='pillow', fps=60)

    
def main():
    
    init_values = pd.read_pickle('init_values.pkl')
    data = pd.read_pickle('sim_values.pkl')
    const_df = pd.read_pickle('consts.pkl')
    
    box_shape = const_df['box_shape'][0]
    steps = const_df['steps'][0]
    frames = steps + 1
    N = const_df['N'][0]
    
    verbose = True
    update_freq = 5 # Number of progress updates e.g. every 20% for 5 

    t1 = time.time()
    
    animate(init_values, data, box_shape, frames, N, t1, verbose, update_freq)
    
    t2 = time.time()
    total_t = t2 - t1
    
    if verbose:
        if total_t > 60:
            long_time = str(datetime.timedelta(seconds=total_t))
            print(f'Time taken for simulation: {long_time}')
        else:
            print(f'Time taken for simulation: {total_t:.2f}s')
        

if __name__ == '__main__':
    main()
    