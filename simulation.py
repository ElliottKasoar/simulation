#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:49:50 2020

@author: Elliott
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle


D = 2 # Number of dimensions

class Particle:

    def __init__(self, r=np.zeros(D), v=np.zeros(D), im=1.0, rad=0.1):
            self.r = r # Position
            self.v = v # Velocity
            self.im = im # Inverse maass
            self.rad = rad # Radius

    # Linear momentum
    def momentum(self):
        return self.v / self.im

    # Linear kinetic energy
    def KE(self):
        return (0.5 * self.v**2) / self.im  

    # Area of particle
    def area(self):
        return pi * self.rad ** 2
    
    # Check if particles overlap
    #Returns True if particle overlaps with existing particles, else False
    def check_overlap(self, particles):
    
        no_overlap = True # True if there is no overlap
        
        for i in range(len(particles)):
            dist = np.linalg.norm(self.r - particles[i].r)
            if (dist < self.rad + particles[i].rad):
                print("Overlap!")
                no_overlap = False
         
        overlap = not(no_overlap)
        return overlap


# Create list of particles. 
# Currently position and velocity random. Particles are checked to not overlap
def create_particles(N):

    particles = []
    
    # Need to change position so can't be partially outside of box
    # Also consider limits on size/velocity
    
    for i in range(N):
        
        r = np.random.rand(D)
        v = np.random.rand(D)
        particle = Particle(r,v)
        
        count = 0
        
        while particle.check_overlap(particles):
            r = np.random.rand(D)
            v = np.random.rand(D)
            particle = Particle(r,v)
            count += 1
            if (count == 10):
                print("Unable to place new particle")
                break
        
        if count < 10:
            particles.append(particle)
        
    return particles


#Plot particles as circles
def plot(particles):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    
    for i in range(len(particles)):
        ax.add_patch(Circle(xy=particles[i].r, radius=particles[i].rad))
    
    # circle1 = Circle(xy=particles[0].r, radius=particles[0].rad)
    # circle2 = Circle(xy=particles[1].r, radius=particles[1].rad)
    # ax.add_patch(circle1)
    # ax.add_patch(circle2)
    
    
def main():
    N = 10 # Number of particles
    # rand = np.random.random()
    particles = create_particles(N)
    print(particles[0].r)
    # print(particles[1].r)
    plot(particles)
    

if __name__ == '__main__':
    main()


