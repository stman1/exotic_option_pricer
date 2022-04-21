# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 08:49:25 2022

@author: Stefan Mangold
"""

from numpy import *
import pandas as pd
from simulation import simulate_path, simulate_path_moment_matched
from BlackScholes import BS
from Contracts import EuropeanOption, OptionType

def test_martingale_property_asset_price_path(s0, drift, vol, timesteps, antithetic):
    
    n_sims = [1000, 10000, 100000, 1000000]
    
    horizons = [0.1, 0.5, 1, 5, 10] # in years
    
    res = []    
    for n in n_sims:
        new_num_sim_list = []
        for h in horizons:
            asset_paths = simulate_path(s0, drift, vol, h, int(timesteps * h), n, antithetic)
            expected_asset_price = exp(-drift * h) * mean(asset_paths[-1])
            new_num_sim_list.append(expected_asset_price)
        res.append(new_num_sim_list)
    return res

def test_euler_maruyama(s0, strike, drift, vol, time_to_expiry, timesteps, antithetic, vol_flag, ts_flag):
    n_sims = 100000
    
    vol_s = [v/100 for v in range(1, 83, 3)]
    strike_s = [st for st in range(25, 525, 25)]
    spot_s = [sp for sp in range(25, 525, 25)] 
    timesteps_s = [252, 504, 756, 1008, 2016, 4032, 8064, 16084] 
    horizons_s = [0.1, 0.5, 1, 5, 10] # in years
    
    
    my_bs = BS(s0, strike, drift, time_to_expiry, vol)
    # European option by analytical BlackScholes
    if (vol_flag):
        vol_res = [['vol', 'BS Call', 'MC Call', 'Call price difference', 'BS Put', 'MC Put', 'Put price difference' ]]
        for v in vol_s:
            my_bs.volatility = v
            asset_paths = simulate_path(s0, drift, v, time_to_expiry, timesteps, n_sims, antithetic)
            my_mc = EuropeanOption(asset_paths, OptionType.CALL, strike, drift, time_to_expiry)
            bscall = my_bs._price()[0]; mccall =  my_mc.call_price
            bsput =  my_bs._price()[1]; mcput =  my_mc.put_price
            vol_res.append([v, bscall, mccall, bscall - mccall, bsput, mcput, bsput - mcput])
    
        vol_frame = pd.DataFrame(vol_res)
        vol_frame.columns = vol_frame.iloc[0]
        vol_frame = vol_frame[1:] 
    
    if (ts_flag):
        ts_vol = 0.2
        my_bs.volatility = ts_vol
        
        ts_res = [['Time Steps', 'BS Call', 'MC Call', 'Call price difference', 'BS Put', 'MC Put', 'Put price difference' ]]
        for ts in timesteps_s:
            asset_paths = simulate_path(s0, drift, ts_vol, time_to_expiry, ts, n_sims, antithetic)
            my_mc = EuropeanOption(asset_paths, OptionType.CALL, strike, drift, time_to_expiry)
            bscall = my_bs._price()[0]; mccall =  my_mc.call_price
            bsput =  my_bs._price()[1]; mcput =  my_mc.put_price
            ts_res.append([ts, bscall, mccall, bscall - mccall, bsput, mcput, bsput - mcput])
    
        ts_frame = pd.DataFrame(ts_res)
        ts_frame.columns = ts_frame.iloc[0]
        ts_frame = ts_frame[1:] 
    
    
    return ts_frame
    
    
