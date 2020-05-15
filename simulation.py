#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:49:50 2020

@author: Elliott
"""

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
# from matplotlib import animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


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
def create_particle(box_shape, max_speed):

    mass = 1.0 
    rad = 0.5
    
    r = np.array(())
    
    for i in range(D):        
        r = np.append(r, np.random.uniform(rad, box_shape[i] - rad))
    
    v = np.random.uniform(-max_speed, max_speed, D)
    
    return Particle(r, v, mass, rad)


# Create list of particles. 
# Currently position and velocity random. Particles are checked to not overlap
def create_particles_list(N, box_shape, max_speed):
    
    particles = []
    
    for i in range(N):
        
        particle = create_particle(box_shape, max_speed)
        count = 0
        
        while particle.check_overlap(particles):
            particle = create_particle(box_shape, max_speed)
            count += 1
            if (count == 10):
                print("Unable to place new particle. Consider decreasing N")
                break
            
        if count < 10:
            particles.append(particle)
    
    return particles


# Collide two particles
def wall_collide(particles, box_shape):
    
    #Check walls first
    for i in range(len(particles)):
        if particles[i].r[0] < particles[i].rad:
            particles[i].r[0] = particles[i].rad
            particles[i].v[0] = -particles[i].v[0]
            
        if particles[i].r[0] > (box_shape[0] - particles[i].rad):
            particles[i].r[0] = box_shape[0] - particles[i].rad
            particles[i].v[0] = -particles[i].v[0]

        if particles[i].r[1] < particles[i].rad:
            particles[i].r[1] = particles[i].rad
            particles[i].v[1] = -particles[i].v[1]

        if particles[i].r[1] > (box_shape[1] - particles[i].rad):
            particles[i].r[1] = box_shape[1] - particles[i].rad
            particles[i].v[1] = -particles[i].v[1]


# Collide two particles in 1/2D, using ZMF to calculate new velocities
def particle_collide(p_1, p_2):
    
    # Must be np array to perform correct operations
    vzm = np.add((p_1.v*p_1.mass), (p_2.v*p_2.mass))
    vzm /= p_1.mass + p_2.mass
    
    p_1.v -= vzm
    p_2.v -= vzm
    
    direction = np.subtract(p_1.r, p_2.r) 
    direction_mag = np.linalg.norm(direction)
    direction_norm = direction / direction_mag 
    
    u_1 = np.linalg.norm(p_1.v)
    u_2 = np.linalg.norm(p_2.v)
    
    p_1.v = direction_norm * u_1
    p_2.v = direction_norm * -u_2
    
    p_1.v += vzm
    p_2.v += vzm	 


# Check which particles are colliding and call collide function where relevant
def check_collision(particles, box_shape):
    
    for i in range(len(particles)):
        for j in range(i+1, len(particles)):
                if particles[i].check_overlap([particles[j]]):
                    particle_collide(particles[i], particles[j])
       

# Update particle states
def update_states(particles, N, dt):
    
    for i in range(N):
        # print("Initial position: ", particles[i].r)
        particles[i].r += particles[i].v * dt
        # print("Updated position: ", particles[i].r)
    
    return particles


# Function for animation to update each frame
def update_animation(frame, particles, box_shape, ax, N, dt):
    
    circles = []
    
    for i in range(len(particles)):
        circles.append(Circle(xy=particles[i].r, radius=particles[i].rad))
        ax.add_patch(Circle(xy=particles[i].r, radius=particles[i].rad))
    
    check_collision(particles, box_shape)
    wall_collide(particles, box_shape)
    update_states(particles, N, dt)
    
    return circles


# Set up axes for animation, creates animation and saves
def create_animation(particles, box_shape, N, steps, dt):
    
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
    
    anim = FuncAnimation(fig, update_animation,
                         fargs=(particles, box_shape, ax, N, dt), 
                         frames=steps, interval=2, blit=True)
    
    anim.save('test.gif', writer='imagemagick', fps=30)


#Plot particles as circles
# Currently ununsed (create_animation instead) but can be used to view a frame
def plot_box(particles, box_shape):
    
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
    
    for i in range(len(particles)):
        ax.add_patch(Circle(xy=particles[i].r, radius=particles[i].rad))
    
    plt.show(fig)
    plt.close(fig)


# Find total KE in system
def KE_tot(particles):
    
    KE = 0
    
    for i in range(len(particles)):
        KE += particles[i].KE()
    
    return KE


# Find mass of smallest particle
def min_mass(particles):
    
    # current_min = particledt = 20s[0].mass
    
    # for i in range(len(particles)):
    #     if particles[i].mass < current_min:
    #         current_min = particles[i].mass
    
    return min(particle.mass for particle in particles)
    # return current_min


# Find radius of smallest particle
def extreme_val(particles, key, minimum=True, index=None):
       
    if minimum:
        return min(abs(key(particle)) for particle in particles)
    else:
        return max(abs(key(particle)) for particle in particles)


# Find maximum possible velocity for single particle, if given all initial KE
def max_v(particles):
    
    max_KE = KE_tot(particles)
    mass = extreme_val(particles, 'mass', minimum=True)
    v = sqrt(2 * max_KE / mass)
    
    return v


# Check maximum possible distance a particle could travel in time step
def max_dist(particles, dt):
    
    v = max_v(particles)
    dist = v * dt
    return dist


# Checks likelihood particles will pass through each other
def speed_check(particles, dt):
    
    dist = max_dist(particles, dt)
    rad = extreme_val(particles, 'rad', minimum=True)

    if dist > rad:
        warning = "Warning: particles may pass through each other. " + \
        "Consider using a smaller dt or max_speed." 
        print(warning)
    else:
        print("Speeds OK")


def main():
    
    N = 2 # Number of particles
    steps = 250 # Number of time steps
    dt = 0.01 # Size of time step
    max_speed = 10
    box_shape = [10, 10]
    
    particles = create_particles_list(N, box_shape, max_speed)    
    
    speed_check(particles, dt)
    initial_KE = KE_tot(particles)
    
    create_animation(particles, box_shape, N, steps, dt)
    
    final_KE = KE_tot(particles)
    KE_change = final_KE - initial_KE
    KE_percent_change = (100 * KE_change) / initial_KE
    
    print(f'The kinetic energy change was {KE_change:.2e} \
          ({KE_percent_change:.2e} %)')
    
    # print("Intial KE: ", initial_KE)
    # print("Final KE: ",  final_KE)
    # print("KE change :", KE_change, "(", KE_percent_change, "%)")
    
    # Plotting frames:
    # plot_box(particles, box_shape)
    
    # for i in range(steps):
        # check_collision(particles, box_shape)
        # wall_collide(particles, box_shape)
        # update_states(particles, N, dt)
        # plot_box(particles, box_shape)

if __name__ == '__main__':
    main()