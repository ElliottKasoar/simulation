#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# To do list:
# Limits on inputs e.g. negative mass/radius?
# Figure out defaults
# Change how params passed?
# Specify initial conditions?
# Seperate simulation and gif making (via saving to txt file?)


import numpy as np
import pandas as pd
from math import pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from operator import attrgetter
import time
import datetime
from scipy.constants import epsilon_0


class Particle:
    
    def __init__(self, r=np.zeros(3), u=np.zeros(3), mass=1.0, rad=0.5,
                 charge=0, colour='tab:blue', v=np.zeros(3), a=np.zeros(3)):
        
        self.r = r # Position
        self.u = u # Velocity at start of time step
        self.v = v # Velocity at end of time step
        self.a = a # Acceleration
        self.mass = mass # Inverse mass
        self.rad = rad # Radius
        self.charge = charge # Radius
        self.colour = colour # Colour of particle when plotting
    
    # Linear momentum at start of time step (usually np array)
    def momentum(self):
        return self.u * self.mass
    
    # Speed of particles at start of time step
    def speed(self):
        return np.linalg.norm(self.u)
    
    # Linear kinetic energy at stat of time step
    def KE(self):
        return 0.5 * self.mass * np.sum(self.u**2)

    # Gravitational potential energy
    def GPE(self, g):
        return self.mass * g * self.r[1]


    def EPE(self, particles):
        
        ke =  1 / ( 4 * pi * epsilon_0)
        E = 0
    
        for i in range(len(particles)):
            
            direction = np.subtract(self.r, particles[i].r) 
            direction_mag = np.linalg.norm(direction)    
            
            E += ke * self.charge * particles[i].charge / direction_mag
        
        return E
    
    
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
    def update_state(self, dt):
    
        self.r += 0.5 * np.add(self.u, self.v) * dt
        self.u = self.v

    def plot_circle(self, ax):
        circle = Circle(xy=self.r, radius=self.rad, color=self.colour)
        # print(self.colour)
        ax.add_patch(circle)
        return circle


# F = dP/dt => dv = F dt / m
def force_particle(particle, particles, D, dt, grav, g, B_field, B, q_int, N):
    
    F =  np.zeros(3)
    
    if grav:
        F[D-1] += -g  * particle.mass
    
    if B_field:
        F += particle.charge * np.cross(particle.u, B)
    
    if q_int:
       
       ke =  1 / ( 4 * pi * epsilon_0)
       
       for i in range(N-1):
           
           direction = np.subtract(particle.r, particles[i].r) 
           direction_mag = np.linalg.norm(direction)
           direction_norm = direction / direction_mag
           
           r_sq = np.sum(np.square(direction))
           F_mag = ke * particle.charge * particles[i].charge / r_sq
           
           F += direction_norm * F_mag
    
    
    # print(F)
    # print(particle.mass)
        
    particle.a = np.divide(F, particle.mass)
    
    dv = np.multiply(particle.a, dt)
    
    # print(dv)
    
    particle.v = np.add(particle.u, dv)
    

# Collide particle with wall
def wall_collide(particle, box_shape):
    
    if particle.r[0] < particle.rad:
        # particle.r[0] = particle.rad
        particle.u[0] = -particle.u[0]
    
    if particle.r[0] > (box_shape[0] - particle.rad):
        # particle.r[0] = box_shape[0] - particle.rad
        particle.u[0] = -particle.u[0]
    
    if particle.r[1] < particle.rad:
        # particle.r[1] = particle.rad
        particle.u[1] = -particle.u[1]
    
    if particle.r[1] > (box_shape[1] - particle.rad):
        # separticlelf.r[1] = box_shape[1] - particle.rad
        particle.u[1] = -particle.u[1]


# Collide two particles in 1/2D, using ZMF to calculate new velocities
def particle_collide(p1, p2):
    
    # Calculuate ZMF velocity
    # Must be np arrays to perform correct operations
    vzm = np.add((p1.u * p1.mass), (p2.u * p2.mass))
    vzm /= p1.mass + p2.mass
    
    # Velocities in ZMF:
    p1_v = np.subtract(p1.u, vzm)
    p2_v = np.subtract(p2.u, vzm)
    
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
    p1.u = p1_v + vzm
    p2.u = p2_v + vzm


# Check which particles are colliding and call collide function where relevant
def check_collision(particles, box_shape, N):
    
    for i in range(N):
        for j in range(i+1, N):
                if particles[i].check_overlap([particles[j]]):
                    particle_collide(particles[i], particles[j])


# Function for animation to update each frame
def update_particles(frame, particles, D, box_shape, N, dt, update_freq, steps,
                t1, verbose, grav, g, B_field, B, q_int):
    
    check_collision(particles, box_shape, N)
    
    for i in range(N):
        
        wall_collide(particles[i], box_shape)
        
        other_particles = particles[:i] + particles[i+1:]
        force_particle(particles[i], other_particles, D, dt, grav, g, B_field,
                       B, q_int, N)
        
        particles[i].update_state(dt)
        
    
    if verbose:
        frame_update = np.linspace(1, update_freq-1, update_freq-1)
        frame_update *= steps//update_freq
        
        if frame in frame_update:
            t2 = time.time()
            total_t = t2 - t1
            long_time = str(datetime.timedelta(seconds=total_t))
            print(f'Simulation progress: {(100 * frame/steps):.1f}%. '
                  f'Current time taken for simulation: {long_time}')


def run_sim(particles, D, box_shape, N, dt, update_freq, steps, t1, verbose,
            grav, g, B_field, B, q_int):
    
    x = np.arange(0, N)
    r_lst = ["r" + num for num in x.astype(str)]
    u_lst = ["u" + num for num in x.astype(str)]
    
    columns = [None]*(N*2) 
    columns[::2] = r_lst
    columns[1::2] = u_lst    
    
    sim_df = pd.DataFrame(columns=columns, index=list(range(steps+1)))
    
    for i in range(N):
        r_name = 'r' + str(i)
        u_name = 'u' + str(i)
        sim_df.iloc[0][r_name] = particles[i].r.copy()
        sim_df.iloc[0][u_name] = particles[i].u.copy()
                
    for i in range(steps):
        
        update_particles(i, particles, D, box_shape, N, dt, update_freq, steps,
                         t1, verbose, grav, g, B_field, B, q_int)
        
        for j in range(N):
            r_name = 'r' + str(j)
            u_name = 'u' + str(j)
            sim_df.iloc[i+1][r_name] = particles[j].r.copy()
            sim_df.iloc[i+1][u_name] = particles[j].u.copy()
    
    
    sim_df.to_pickle('sim_values.pkl')


# Function for animation to update each frame
def update_anim(frame, sim_df, particles, D, box_shape, ax, N, dt, update_freq,
                steps, t1, verbose, save_sim, grav, g, B_field, B, q_int):
    
    circles = []
    
    if (frame==0):
        
        for i in range(N):
            circles.append(particles[i].plot_circle(ax))
        
    else:
    
        check_collision(particles, box_shape, N)
        
        for i in range(N):
            
            wall_collide(particles[i], box_shape)
            
            other_particles = particles[:i] + particles[i+1:]
            force_particle(particles[i], other_particles, D, dt, grav, g, B_field,
                           B, q_int, N)
            
            particles[i].update_state(dt)
        
        if verbose:
            frame_update = np.linspace(1, update_freq-1, update_freq-1)
            frame_update *= steps//update_freq
            
            if frame in frame_update:
                t2 = time.time()
                total_t = t2 - t1
                long_time = str(datetime.timedelta(seconds=total_t))
                print(f'Simulation progress: {(100 * frame/steps):.1f}%. '
                      f'Current time taken for simulation: {long_time}')
        
                
        for i in range(N):
           
            if (save_sim):
                r_name = 'r' + str(i)
                u_name = 'u' + str(i)
                sim_df.iloc[frame][r_name] = particles[i].r.copy()
                sim_df.iloc[frame][u_name] = particles[i].u.copy()   
            
            circles.append(particles[i].plot_circle(ax))
                                
    return circles


# Set up axes for animation, creates animation and saves
def create_anim(particles, D, box_shape, N, steps, dt, update_freq, t1,
                verbose, save_sim, grav, g, B_field, B, q_int):
    
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
    
    if (save_sim):
        
        x = np.arange(0, N)
        r_lst = ["r" + num for num in x.astype(str)]
        u_lst = ["u" + num for num in x.astype(str)]
    
        columns = [None]*(N*2) 
        columns[::2] = r_lst
        columns[1::2] = u_lst    
        
        sim_df = pd.DataFrame(columns=columns, index=list(range(steps+1)))

        for i in range(N):
            r_name = 'r' + str(i)
            u_name = 'u' + str(i)
            sim_df.iloc[0][r_name] = particles[i].r.copy()
            sim_df.iloc[0][u_name] = particles[i].u.copy()
            
    else:
        sim_df = pd.DataFrame()
    
    anim = FuncAnimation(fig, update_anim, frames=(steps+1), interval=2,
                         blit=True,
                         fargs=(sim_df, particles, D, box_shape, ax, N, dt,
                                update_freq, steps, t1, verbose, save_sim,
                                grav, g, B_field, B, q_int))
    
    anim.save('particles_set.gif', writer='pillow', fps=30)
    
    if (save_sim):
        sim_df.to_pickle('sim_values.pkl')


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
    fig.savefig('Box.pdf', dps=1000)
    plt.close(fig)


# Find total KE in system
def calc_E_tot(particles, N, grav, g, q_int):
    
    KE = 0
    GPE = 0
    EPE = 0
    
    for i in range(N):
        KE += particles[i].KE()
        
    if grav:
        
        for i in range(N):
            GPE += particles[i].GPE(g)
        
        E_tot = KE + GPE
    
    if q_int:
        
        for i in range(N):
        
            other_particles = particles[:i] + particles[i+1:]
            EPE += particles[i].EPE(other_particles)
        
    E_tot = KE + GPE + EPE 
    
    return E_tot


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
def calc_max_v(particles, N, grav, g, q_int):
    
    max_KE = abs(calc_E_tot(particles, N, grav, g, q_int))    
    min_mass = calc_extreme_val(particles, 'mass', minimum=True)
    max_v = sqrt(2 * max_KE / min_mass)
    
    return max_v


# Check maximum possible distance a particle could travel in time step
def calc_max_dist(particles, dt, N, grav, g, q_int):
    
    v = calc_max_v(particles, N, grav, g, q_int)
    max_dist = v * dt
    
    return max_dist


def calc_expt_av_dist(vel_range, dt):
    
    # Approx
    # sample_num = 10000000
    # a = np.random.uniform(vel_range[0], vel_range[1], sample_num)
    # b = np.random.uniform(vel_range[0], vel_range[1], sample_num)
    # v_av = np.mean(np.sqrt(np.add(np.square(a), np.square(b))))
    
    max_speed = abs(max([vel_range[0], vel_range[1]], key=abs))
    
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
def speed_check(particles, dt, vel_range, N, grav, g, q_int):
    
    max_dist = calc_max_dist(particles, dt, N, grav, g, q_int)
    expt_av_dist = calc_expt_av_dist(vel_range, dt)
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


# Creates single particle
# Defines mass, radius, position, velocity, charge and colour from ranges given
# Colour based on masses (all same or darker if heavier)
def create_rand_particle(D, box_shape, vel_range, mass_range, rad_range,
                         q_range, e):
    
    # Mass of particle and colour (if variable masses, darker when heavier)
    if (mass_range[0]==mass_range[1]):
        mass = mass_range[0]
        colour = 'tab:blue'
    else:
        mass = np.random.uniform(mass_range[0], mass_range[1])
        colour = (0,0, 1-(mass/mass_range[1]))
        
    # Radius of particle
    if (rad_range[0]==rad_range[1]):
        rad = rad_range[0]
    else:
        rad = np.random.uniform(rad_range[0], rad_range[1])
   
    # Position of particle
    r = np.array(())
    
    #Position within box
    for i in range(D):
        
        if (box_shape[i] - 2*rad < 0):
            print("Warning: particle is too large for box")
        r = np.append(r, np.random.uniform(rad, box_shape[i] - rad))
    
    # Position in dimensions not defined by box e.g. z coordinate for 2D box
    for i in range(D, 3):
        r = np.append(r, np.array(box_shape[i]/2))
    
    # Velocity of particle
    u = np.random.uniform(vel_range[0], vel_range[1], D)
    u = np.append(u, np.zeros(3-D))
    
    # Charge of particle
    if (q_range[0]==q_range[1]):
        charge = q_range[0] * e
    elif np.random.random() < 0.5:
        charge = q_range[0] * e
    else:
        charge = q_range[1] * e
    
    # Random half int:
    # charge = round(np.random.uniform(q_range[0], q_range[1]) * 2) * e / 2
    
    return Particle(r, u, mass, rad, charge, colour)


# Creates single particle from hard-coded values
# Sets mass, radius, position, velocity, charge and colour using dataframe
# Colour based on masses (all same or darker if heavier)
def create_set_particle(D, df, i, e):
    
    mass = df.iloc[i]['mass']
    
    if (df['mass'].max() == df['mass'].min()):
        colour = 'tab:blue'
    else:
        colour = (0,0, 1-(mass/df['mass'].max()))
    
    rad = df.iloc[i]['rad']

    r = df.iloc[i]['r']
    
    u = df.iloc[i]['u']

    charge = df.iloc[i]['charge'] * e
    
    return Particle(r, u, mass, rad, charge, colour)


# Create list of particles. 
# Currently position and velocity random. Particles are checked to not overlap
def create_particles_list(D, N, box_shape, vel_range, mass_range, rad_range,
                          q_range, e, rand_init, read_init, particle_df):
    
    particles = []
    
    for i in range(N):
        
        if (rand_init):
            
            particle = create_rand_particle(D, box_shape, vel_range,
                                            mass_range, rad_range, q_range, e)
            
            count = 0
            
            while particle.check_overlap(particles):
            
                particle = create_rand_particle(D, box_shape, vel_range,
                                                mass_range, rad_range, q_range,
                                                e)
                count += 1
            
                if (count == 10):
                    print("Unable to place new particle. Consider decreasing N "
                      "or increasing box size")
                    break
        
            if count < 10:
                particles.append(particle)
            
        else:
            particle = create_set_particle(D, particle_df, i, e)
            particles.append(particle)
        
        if (not read_init):
            particle_df.loc[i] = [particle.r, particle.u, particle.mass, 
                                particle.rad, particle.charge, particle.colour]
    
    if (not read_init):    
        particle_df.to_pickle('init_values.pkl')
    
    return particles


def main():
    
    N = 4 # Number of particles. Default = 6
    steps = 5000 - 1 # Number of time steps. Default = 599 (includes frame 0)
    dt = 1/300 # Size of time step.  Default = 1/300
    box_shape = [1, 1, 1] # Size of box. Default = [1,1, 1]
    D = 2 # Number of dimensions. Default = 2 (works for 1, should work for 3?)
    
    anim = False
    save_sim = True
    rand_init = True
    read_init = False
    
    verbose = True
    update_freq = 5 # Number of progress updates e.g. every 20% for 5
    
    grav = False
    g = 9.81
    
    B_field = False
    B = np.array([0, 0, 1e-7])
    
    q_int = False
    e = 1.60e-19
    
    # Min and max values for velocity in each direction:
    vel_range = [-1.0, 1.0] # Default = [-2.0, 2.0]
    
    # Min and max values for range
    mass_range = [1.0, 1.0] # Default = [1.0, 10.0]
    
    # Min and max values for radius    
    rad_range = [0.1, 0.1] # Default = [0.02, 0.1]

    # Min and max values for charge
    q_range = [-1, 1] # Default = [-1, 1] 
    
    const_df = pd.DataFrame(columns=['N', 'steps', 'dt', 'box_shape', 'D',
                                        'grav', 'g', 'B_field', 'B', 'q_int',
                                        'e'])
    
    const_df.loc[0] = N, steps, dt, box_shape, D, grav, g, B_field, B, q_int, e
    const_df.to_pickle('consts.pkl')
    
    # Read or create dataframe to save inital conditions
    if (read_init and not rand_init):
        print("Using initial conditions saved")
        particle_df = pd.read_pickle('init_values.pkl')
    else:
        particle_df = pd.DataFrame(columns=['r', 'u', 'mass', 'rad', 'charge',
                                        'colour'])
    
    
    # Define inital conditions if not reading from file or generating randomly
    if (not read_init and not rand_init):
         
        print("Using initial conditions specified")
        r = [np.array([0.1, 0.2, 0.5]),
             np.array([0.6, 0.8, 0.5]),
             np.array([0.7, 0.4, 0.5])]
        
        u = [np.array([1.0, 1.0, 0.0]),
             np.array([1.0, -1.0, 0.0]),
             np.array([-0.5, 0.5, 0.0])]
        
        mass = [3.0, 2.0, 1.0]
        
        rad = [0.02, 0.05, 0.1]
        
        charge = [0.0, 0.0, 0.0]
        
        colour = [None, None, None]
        
        particle_df.loc[0] = r[0], u[0], mass[0], rad[0], charge[0], colour[0]
        particle_df.loc[1] = r[1], u[1], mass[1], rad[1], charge[1], colour[1]
        particle_df.loc[2] = r[2], u[2], mass[2], rad[2], charge[2], colour[2]

    
    particles = create_particles_list(D, N, box_shape, vel_range, mass_range,
                                      rad_range, q_range, e, rand_init,
                                      read_init, particle_df)    
    
    N = len(particles)
    
    if verbose:
        # Check likelihood of particles passing through each other
        speed_check(particles, dt, vel_range, N, grav, g, q_int)
        
        # Check initial KE so can track and confirm conservation
        initial_E = calc_E_tot(particles, N, grav, g, q_int)
    
    t1 = time.time()
    
    # Create animation of simulation
    if (anim):
        create_anim(particles, D, box_shape, N, steps, dt, update_freq, t1,
                verbose, save_sim, grav, g, B_field, B, q_int)
    elif (save_sim):
        run_sim(particles, D, box_shape, N, dt, update_freq, steps, t1, 
                verbose, grav, g, B_field, B, q_int)
    else:
        print("Please choose to create animation and/or save simulation data")
        
    # print(particles[0].r, particles[0].u)
    # print(particles[0].KE(), particles[0].GPE(g))
    
    if verbose:
        t2 = time.time()
        total_t = t2 - t1
        if total_t > 60:
            long_time = str(datetime.timedelta(seconds=total_t))
            print(f'Time taken for simulation: {long_time}')
        else:
            print(f'Time taken for simulation: {total_t:.2f}s')
        
        final_E = calc_E_tot(particles, N, grav, g, q_int)
        E_change = final_E - initial_E
        E_percent_change = (100 * E_change) / initial_E
        E_txt = (f'The energy change was {E_change:.2e} '
                  f'({E_percent_change:.2e} %)')
        print(E_txt)
    

if __name__ == '__main__':
    main()