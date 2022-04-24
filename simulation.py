# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:29:14 2022

@author: Kannan Singaravelu
"""

import numpy as np

def antithetic_rv(num_rvs):
    num_rvs = int(num_rvs)
    if num_rvs % 2 == 0:    
        sn_1 = np.random.standard_normal(num_rvs//2)                 
        sn_2 = -sn_1
        w = np.concatenate((sn_1, sn_2), axis=0)
    else:
        sn_1 = np.random.standard_normal((num_rvs-1)//2)                 
        sn_2 = -sn_1
        w = np.concatenate((sn_1, sn_2), axis=0)
        w.append(np.random.standard_normal(1))        
    return w
    

def moment_matching(mean, standard_deviation, num_paths, num_observations):
    
    np.random.seed(10000) 
    num_paths = int(num_paths); standard_deviation = int(standard_deviation)
        
    # antithetic sampling
    phi = np.random.normal(mean, standard_deviation, [num_paths//2, num_observations]).astype('f')        
    sample = np.concatenate((phi, -phi), axis=0)

    if num_paths % 2 == 1:
        last_path = np.random.normal(mean, standard_deviation, [1, num_observations])
        sample = np.concatenate((sample, last_path), axis = 0)
        
    
    # making sure that samples from normal have mean and variance match the inputs
    for obs in range(0, num_observations):
        sample[:, obs] = sample[:, obs] - np.mean(sample[:, obs]) / np.std(sample[:, obs])
        
    return sample


def simulate_path(s0, mu, sigma, horizon, timesteps, n_sims, antithetic):
    
    # Set the random seed for reproducibility
    # Same seed leads to the same set of random values
    #np.random.seed(10000) 

    # Read parameters
    S0 = s0             # initial spot level
    r = mu              # mu = rf in risk neutral framework 
    T = horizon         # time horizion
    t = timesteps       # number of time steps
    n = n_sims          # number of simulations
    ant = antithetic    # switch antithetic variables
    
    # Define dt
    dt = T/t        # length of time interval  
    
    # Simulating 'n' asset price paths with 't' timesteps
    S = np.zeros((t, n))
    S[0] = S0

    for i in range(0, t-1):
        if ant:
            w = antithetic_rv(n)
        else:
            w = np.random.standard_normal(n)                                                         
        S[i+1] = S[i] * (1 + r * dt + sigma * np.sqrt(dt) * w)               # vectorized operation per timesteps
        #S[i+1] = S[i] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * w)  # alternate form
        
    return S


def simulate_path_moment_matched(s0, mu, sigma, horizon, timesteps, n_sims):
    
    # Set the random seed for reproducibility
    # Same seed leads to the same set of random values
    np.random.seed(10000) 

    # Read parameters
    S0 = s0             # initial spot level
    r = mu              # mu = rf in risk neutral framework 
    T = horizon         # time horizion
    t = timesteps       # number of time steps
    n = n_sims          # number of simulations
    
    # Define dt
    dt = T/t        # length of time interval  
    
    # Simulating 'n' asset price paths with 't' timesteps
    S = np.zeros((t, n))
    S[0] = S0

    # generate standard normal variates
    
    w = moment_matching(0., 1., n_sims, timesteps)

    for i in range(0, t-1):                                                       
        S[i+1] = S[i] * (1 + r * dt + sigma * np.sqrt(dt) * w[:, i])               # vectorized operation per timesteps 
    return S