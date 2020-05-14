#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:49:50 2020

@author: Elliott
"""

import numpy as np
from math import pi

D = 2 # Number of dimensions

# Keep static methods or not..?
class Particle:

    def __init__(self, r=np.zeros(D), v=np.zeros(D), im=1.0, rad=1.0):
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
    
    def checkSpace(self):
        print("CHECK")
        return True

def create_particles(N):
    
    particles = []
    
    for i in range(N):
        r = np.random.rand(D)
        v = np.random.rand(D)
        particle = Particle(r,v)
        if particle.checkSpace():
            particles.append(particle)
        
    return particles

    
def main():
    N = 2 # Number of particles
    # rand = np.random.random()
    particles = create_particles(N)
    print(particles[0].momentum())
    

if __name__ == '__main__':
    main()






