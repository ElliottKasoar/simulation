#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:49:50 2020

@author: Elliott
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle


D = 2 # Number of dimensions

# Keep static methods or not..?
class Particle:

    def __init__(self, r=np.zeros(D), v=np.zeros(D), im=1.0, rad=0.05):
            self.r = r # Position
            self.v = v # Velocity
            self.im = im # Inverse maass
            self.rad = rad # Radius

    def momentum(self):
        return self.particleMomentum(self.v, self.im)
    
    @staticmethod
    def particleMomentum(v, im):
        return v / im

    def KE(self):
        return self.particleKE(self.v, self.im)
    
    @staticmethod
    def particleKE(v, im):
        return (0.5 * v ** 2) / im  

    def area(self):
        return self.particleArea(self.rad)
    
    @staticmethod
    def particleArea(rad):
        return pi * rad ** 2
    
    # Without static method
    def altArea(self):
        return pi * self.rad ** 2
    
    def checkSpace(self, particles):
        
        check = True
        
        for i in range(len(particles)):
            dist = np.linalg.norm(self.r - particles[i].r)
            if (dist < self.rad + particles[i].rad):
                print("Overlap!")
                # check = False
                
        return check

def create_particles(N):
    
    particles = []
    
    # Need to change position so can't be partially outside of box
    # Also consider limits on size/velocity
    # Add some sort of while loop to keep checking particle is ok
    # Also add counter or something to give up it can't create more particles?
    
    for i in range(N):
        r = np.random.rand(D)
        v = np.random.rand(D)
        particle = Particle(r,v)
        if particle.checkSpace(particles):
            particles.append(particle)
        
    return particles


def plot(particles):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    circle1 = Circle(xy=particles[0].r, radius=particles[0].rad)
    circle2 = Circle(xy=particles[1].r, radius=particles[1].rad)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    
def main():
    N = 2 # Number of particles
    # rand = np.random.random()
    particles = create_particles(N)
    print(particles[0].r)
    print(particles[1].r)
    plot(particles)
    

if __name__ == '__main__':
    main()






