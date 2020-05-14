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
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from celluloid import Camera


D = 2 # Number of dimensions

class Particle:
    
    def __init__(self, r=np.zeros(D), v=np.zeros(D), mass=1.0, rad=0.5):
        self.r = r # Position
        self.v = v # Velocity
        self.mass = mass # Inverse mass
        self.rad = rad # Radius
    
    # Linear momentum (usually np array)
    def momentum(self):
        return self.v * self.mass
    
    # Linear kinetic energy
    def KE(self):
        return 0.5 * self.mass * np.sum(self.v**2)
    
    # Area of particle
    def area(self):
        return pi * self.rad**2
    
    # Check if particles overlap
    # Returns True if particle overlaps with existing particles, else False
    def check_overlap(self, particles):
        
        no_overlap = True # True if there is no overlap
        
        for i in range(len(particles)):
            dist = np.linalg.norm(self.r - particles[i].r)
            if (dist < self.rad + particles[i].rad):
                # print("Overlap!")
                no_overlap = False
        
        overlap = not(no_overlap)
        return overlap


# Creates single particle, currently with random position and velocity
# Need to change position so can't be partially outside of box
# Also consider limits on size/velocity
# Limits may be implemented via @property i.e. not here?
def create_particle(box_shape):

    mass = 1.0 
    rad = 0.5
    
    r = np.array(())
    
    for i in range(D):        
        r = np.append(r, np.random.uniform(rad, box_shape[i] - rad))
    
    v = np.random.uniform(-5, 5, D)
    
    return Particle(r, v, mass, rad)


# Create list of particles. 
# Currently position and velocity random. Particles are checked to not overlap
def create_particles_list(N, box_shape):
    
    particles = []
    
    for i in range(N):
        
        particle = create_particle(box_shape)
        count = 0
        
        while particle.check_overlap(particles):
            particle = create_particle(box_shape)
            count += 1
            if (count == 10):
                print("Unable to place new particle. Consider decreasing N")
                break
            
        if count < 10:
            particles.append(particle)
    
    return particles


#C heck which particles are colliding and call collide function where relevant
def check_collision(particles):
    print()


# Collide two particles
def collide(particles):
    print()


# Update particle states
def update_states(particles, N, dt):
    
    for i in range(N):
        # print("Initial position: ", particles[i].r)
        particles[i].r += particles[i].v * dt
        # print("Updated position: ", particles[i].r)
    
    return particles


#Plot particles as circles
def plot_box(particles, box_shape):
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, box_shape[0])
    ax.set_ylim(0, box_shape[1])
    ax.set_aspect('equal')
    
    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
    
    for i in range(len(particles)):
        ax.add_patch(Circle(xy=particles[i].r, radius=particles[i].rad))
    
    plt.show(fig)
    plt.close(fig)
    
    # circle1 = Circle(xy=particles[0].r, radius=particles[0].rad)
    # circle2 = Circle(xy=particles[1].r, radius=particles[1].rad)
    # ax.add_patch(circle1)
    # ax.add_patch(circle2)


def update_animation(frame, particles, box_shape, ax, N, dt):
    
    circles = []
    
    for i in range(len(particles)):
        circles.append(Circle(xy=particles[i].r, radius=particles[i].rad))
        ax.add_patch(Circle(xy=particles[i].r, radius=particles[i].rad))
    
    check_collision(particles)
    collide(particles)
    update_states(particles, N, dt)
    
    return circles

# def init_animation(particles, box_shape):
    
#     circles = []
    
#     for i in range(len(particles)):
#         circles.append(Circle(xy=particles[i].r, radius=particles[i].rad))

#     return


def create_animation(particles, box_shape, N, steps, dt):
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, box_shape[0])
    ax.set_ylim(0, box_shape[1])
    ax.set_aspect('equal')
    
    ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False) # labels along the bottom edge are off
    
    anim = FuncAnimation(fig, update_animation,
                         fargs=(particles, box_shape, ax, N, dt), 
                         frames=steps, interval=2, blit=True)
    
    anim.save('test.gif', writer='imagemagick', fps=30)


def main():
    
    N = 2 # Number of particles
    steps = 100 # Number of time steps
    dt = 0.01 # Size of time step
    box_shape = [10, 10]
    
    particles = create_particles_list(N, box_shape)
    # print(particles[0].r)
    # print(particles[1].r)
    
    # plot_box(particles, box_shape)
    
    create_animation(particles, box_shape, N, steps, dt)
    
    # for i in range(steps):
        # check_collision(particles)
        # collide(particles)
        # particles = update_states(particles, N, dt)
        # plot_box(particles, box_shape)

if __name__ == '__main__':
    main()

