# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 08:49:25 2022

@author: Stefan Mangold
"""
from simulation import simulate_path, simulate_path_moment_matched
from numpy import *

def test_martingale_property_asset_price_path(s0, drift, vol, timesteps, antithetic):
    
    n_sims = [1000, 10000, 100000, 1000000]
    
    horizons = [0.1, 0.5, 1, 5, 10] # in years
    
    res = []    
    for n in n_sims:
        new_num_sim_list = []
        for h in horizons:
            #asset_paths = simulate_path(s0, drift, vol, h, int(timesteps * h), n, antithetic)
            asset_paths = simulate_path_moment_matched(s0, drift, vol, h, int(timesteps * h), n)
            expected_asset_price = exp(-drift * h) * mean(asset_paths[-1])
            new_num_sim_list.append(expected_asset_price)
        res.append(new_num_sim_list)
    return res