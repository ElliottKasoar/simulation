#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:49:50 2020

@author: Elliott
"""

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from operator import attrgetter
import time
import datetime

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
    
    def speed(self):
        return np.linalg.norm(self.v)
    
    # Linear kinetic energy
    def KE(self):
        return 0.5 * self.mass * np.sum(self.v**2)
    
    # Area of particle
    def area(self):
        return pi * self.rad**2
    
    # Check if a particle overlaps with a list of particles
    # Returns True if particle overlaps with any other in list, else False
    # Note: will return True if particle exists in list passed
    def check_overlap(self, particles):
        
        no_overlap = True # True if there is no overlap
        
        for i in range(len(particles)):
            dist = np.linalg.norm(self.r - particles[i].r)
            if (dist < self.rad + particles[i].rad):
                # print("Overlap!")
                no_overlap = False
        
        overlap = not(no_overlap)
        return overlap
    
    # Update states for next time step. Also handles collisions with walls
    def update_state(self, dt, box_shape):
        
        self.r += self.v * dt
        
        if self.r[0] < self.rad:
            self.r[0] = self.rad
            self.v[0] = -self.v[0]
        
        if self.r[0] > (box_shape[0] - self.rad):
            self.r[0] = box_shape[0] - self.rad
            self.v[0] = -self.v[0]
        
        if self.r[1] < self.rad:
            self.r[1] = self.rad
            self.v[1] = -self.v[1]
        
        if self.r[1] > (box_shape[1] - self.rad):
            self.r[1] = box_shape[1] - self.rad
            self.v[1] = -self.v[1]


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


# Collide two particles in 1/2D, using ZMF to calculate new velocities
def particle_collide(p1, p2):
    
    # Calculuate ZMF velocity
    # Must be np arrays to perform correct operations
    vzm = np.add((p1.v*p1.mass), (p2.v*p2.mass))
    vzm /= p1.mass + p2.mass
    
    # Velocities in ZMF:
    p1_v = np.subtract(p1.v, vzm)
    p2_v = np.subtract(p2.v, vzm)
    
    # Normalised vector connecting centres of particles
    direction = np.subtract(p1.r, p2.r) 
    direction_mag = np.linalg.norm(direction)
    direction_norm = direction / direction_mag
    
    # Magnitude of velocities along direction of line of centres
    u_1 = np.linalg.norm(p1_v)
    u_2 = np.linalg.norm(p2_v)
    p1_v = direction_norm * u_1
    p2_v = direction_norm * (-u_2)
    
    # Update velocities in lab frame
    p1.v = p1_v + vzm
    p2.v = p2_v + vzm


# Check which particles are colliding and call collide function where relevant
def check_collision(particles, box_shape, N):
    
    for i in range(N):
        for j in range(i+1, N):
                if particles[i].check_overlap([particles[j]]):
                    particle_collide(particles[i], particles[j])


# Function for animation to update each frame
def update_anim(frame, particles, box_shape, ax, N, dt, update_freq, steps, t1, verbose):
    
    circles = []
    
    for i in range(N):
        circles.append(Circle(xy=particles[i].r, radius=particles[i].rad))
        ax.add_patch(Circle(xy=particles[i].r, radius=particles[i].rad))
    
    check_collision(particles, box_shape, N)
    
    for i in range(N):
        particles[i].update_state(dt, box_shape)
    
    if verbose:
        frame_update = np.linspace(1, update_freq-1, update_freq-1)
        frame_update *= steps//update_freq
        
        if frame in frame_update:
            t2 = time.time()
            total_t = t2 - t1
            long_time = str(datetime.timedelta(seconds=total_t))
            print(f'Simulation progress: {(100 * frame/steps):.1f}%. '
                  f'Current time taken for simulation: {long_time}')
        
    return circles


# Set up axes for animation, creates animation and saves
def create_anim(particles, box_shape, N, steps, dt, update_freq, t1, verbose):
    
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
    
    anim = FuncAnimation(fig, update_anim, frames=steps, interval=2, blit=True,
                         fargs=(particles, box_shape, ax, N, dt, update_freq,
                                steps, t1, verbose))
    
    anim.save('test2.gif', writer='imagemagick', fps=30)


#Plot particles as circles
# Currently ununsed (create_animation instead) but can be used to view a frame
def plot_box(particles, box_shape, N):
    
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
    
    for i in range(N):
        ax.add_patch(Circle(xy=particles[i].r, radius=particles[i].rad))
    
    plt.show(fig)
    plt.close(fig)


# Find total KE in system
def KE_tot(particles, N):
    
    KE = 0
    
    for i in range(N):
        KE += particles[i].KE()
    
    return KE


# Find minimum or maximum value of attribute from list of particles
# Defaults to finding minimum. 
# Can specify index for attributes in form of arrays e.g. v or r
def calc_extreme_val(particles, attr, minimum=True, index=None):
    
    key = attrgetter(attr)

    if index is not None:
        if minimum:
            return min(abs(key(particle)[index]) for particle in particles)
        else:
            return max(abs(key(particle)[index]) for particle in particles)
    else:
        if minimum:
            return min(abs(key(particle)) for particle in particles)
        else:
            return max(abs(key(particle)) for particle in particles)


# Find maximum possible velocity for single particle, if given all initial KE
def calc_max_v(particles, N):
    
    max_KE = KE_tot(particles, N)
    min_mass = calc_extreme_val(particles, 'mass', minimum=True)
    max_v = sqrt(2 * max_KE / min_mass)
    
    return max_v


# Check maximum possible distance a particle could travel in time step
def calc_max_dist(particles, dt, N):
    
    v = calc_max_v(particles, N)
    max_dist = v * dt
    
    return max_dist


def calc_expt_av_dist(max_speed, dt):
    
    # Approx
    # sample_num = 10000000
    # a = np.random.uniform(-max_speed, max_speed, sample_num)
    # b = np.random.uniform(-max_speed, max_speed, sample_num)
    # v_av = np.mean(np.sqrt(np.add(np.square(a), np.square(b))))
    
    # Actual expression in 2D:
    v_av = max_speed * (sqrt(2) + np.arcsinh((1))) / 3
        
    v_rel = sqrt(2) * v_av
    expected_av_dist = v_rel * dt
    
    return expected_av_dist


def calc_actual_av_dist(particles, dt, N):
    
    total = 0
    for particle in particles:
        total += particle.speed()

    v_av = total / N
    
    v_rel = sqrt(2) * v_av
    actual_av_dist = v_rel * dt
    
    return actual_av_dist


# Checks likelihood particles will pass through each other
def speed_check(particles, dt, max_speed, N):
    
    max_dist = calc_max_dist(particles, dt, N)
    expt_av_dist = calc_expt_av_dist(max_speed, dt)
    actual_av_dist = calc_actual_av_dist(particles, dt, N)
    
    rad = calc_extreme_val(particles, 'rad', minimum=True)

    if actual_av_dist > 2 * rad:
        warning = ("Warning: particles are highly likely to pass through each "
                   "other. Consider using a smaller dt or max_speed.")
        print(warning)
    
    elif expt_av_dist > 2 * rad:
        warning = ("Warning: particles are likely to pass through each other. "
                   "This is expected given the max_speed and dt specified. "
                   "Consider using a smaller dt or max_speed.")
        print(warning)
        
    elif max_dist > 2 * rad:
        warning = ("Warning: particles may pass through each other. "
                   "Consider using a smaller dt or max_speed.")
        print(warning)
    
    else:
        print('max_speed and dt should be OK.')
    
    print(f'Maximum distance that can be travelled: {max_dist:.2f}.')
    print(f'Expected average distance travelled: {expt_av_dist:.2f}.')
    print(f'Actual average distance travelled: {actual_av_dist:.2f}.')
    print(f'Smallest particle radius: {rad:.2f}')


def main():
    
    N = 5 # Number of particles
    steps = 200 # Number of time steps
    dt = 0.01 # Size of time step
    max_speed = 10 # Maximum magnitude of vx and vy
    box_shape = [10, 10] # Shape of box particles are in (x,y)
    update_freq = 5
    verbose = True
    
    particles = create_particles_list(N, box_shape, max_speed)
    
    if verbose:
        # Check likelihood of particles passing through each other
        speed_check(particles, dt, max_speed, N)
        
        # Check initial KE so can track and confirm conservation
        initial_KE = KE_tot(particles, N)
    
    t1 = time.time()
    
    # Create animation of simulation
    create_anim(particles, box_shape, N, steps, dt, update_freq, t1, verbose)
    
    if verbose:
        t2 = time.time()
        total_t = t2 - t1
        if total_t > 60:
            long_time = str(datetime.timedelta(seconds=total_t))
            print(f'Time taken for simulation: {long_time}')
        else:
            print(f'Time taken for simulation: {total_t:.2f}s')
        
        final_KE = KE_tot(particles, N)
        KE_change = final_KE - initial_KE
        KE_percent_change = (100 * KE_change) / initial_KE
        KE_txt = (f'The kinetic energy change was {KE_change:.2e}'
                  f'({KE_percent_change:.2e} %)')
        print(KE_txt)
    

if __name__ == '__main__':
    main()