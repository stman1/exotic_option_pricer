# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 08:49:25 2022

@author: Stefan Mangold
"""

import numpy as np
from numpy import *
import pandas as pd

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from simulation import simulate_path, simulate_path_moment_matched
from BlackScholes import BS
from AnalyticGeometricAsian import ClosedFormGeometricAsian
from AnalyticContinuousLookback import ClosedFormContinuousLookback
from Contracts import AsianOption, LookbackOption, EuropeanOption, PayoffType, OptionType, AveragingType, StrikeType

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

def test_martingale_property_asset_price_path_repeated(s0, drift, vol, timesteps, antithetic, n_reps):
    
    n_sims = [1000, 10000, 100000, 1000000]
    
    horizons = [0.1, 0.5, 1, 5, 10] # in years
    
    aggregator_array = np.zeros((len(n_sims), len(horizons))) 
    
    for r in range(0, n_reps):
        for s_idx, s in enumerate(n_sims):
            for h_idx, h in enumerate(horizons):
                asset_paths = simulate_path(s0, drift, vol, h, int(timesteps * h), s, antithetic)
                aggregator_array[s_idx][h_idx] += exp(-drift * h) * mean(asset_paths[-1])

    return aggregator_array/n_reps


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
    

def test_closed_form_solutions(spot_price, strike_price, risk_free_rate, time_horizon, asset_volatility, timesteps):
    spot_space = linspace(50, 150, 30)
    time_space = linspace(0.001, 5, 30)

    # Closed-form Asian option
    my_geometric_asian = ClosedFormGeometricAsian(spot_price, strike_price, risk_free_rate, time_horizon, asset_volatility, timesteps)
    print(f'Geometric Asian option call and put price: {my_geometric_asian.call_price:0.4f}, {my_geometric_asian.put_price:0.4f}')

    # Black-Scholes formula

    my_black_scholes = BS(spot_price, strike_price, risk_free_rate, time_horizon, asset_volatility)
    print(f'Black-Scholes option call and put price: {my_black_scholes.call_price:0.4f}, {my_black_scholes.put_price:0.4f}')

    # Closed-form Lookback option
    my_lookback = ClosedFormContinuousLookback(spot_price, OptionType.CALL, StrikeType.FIXED, risk_free_rate, time_horizon, asset_volatility, strike_price, spot_price, spot_price)
    print(f'Lookback option call and put price: {my_lookback.call_price:0.4f}, {my_lookback.put_price:0.4f}')

    results = np.zeros((3, 2, len(spot_space), len(time_space)))
    for sp_idx, sp in enumerate(spot_space, 0):
        for ts_idx, ts in enumerate(time_space, 0):
            
            my_black_scholes = BS(sp, strike_price, risk_free_rate, ts, asset_volatility)
            my_geometric_asian = ClosedFormGeometricAsian(sp, strike_price, risk_free_rate, ts, asset_volatility, timesteps)
            my_lookback = ClosedFormContinuousLookback(sp, OptionType.CALL, StrikeType.FLOATING, risk_free_rate, ts, asset_volatility, strike_price, sp, sp)

            results[0, 0, sp_idx, ts_idx] = my_black_scholes.call_price
            results[1, 0, sp_idx, ts_idx] = my_geometric_asian.call_price
            results[2, 0, sp_idx, ts_idx] = my_lookback.call_price
            results[0, 1, sp_idx, ts_idx] = my_black_scholes.put_price
            results[1, 1, sp_idx, ts_idx] = my_geometric_asian.put_price
            results[2, 1, sp_idx, ts_idx] = my_lookback.put_price
        
        
    float_formatter = "{:.2f}".format    
    np.set_printoptions(formatter={'float_kind': float_formatter})



    def plottable_3d_info(df: pd.DataFrame):
        """
        Transform Pandas data into a format that's compatible with
        Matplotlib's surface and wireframe plotting.
        """
        index = df.index
        columns = df.columns

        x, y = np.meshgrid(np.arange(len(columns)), np.arange(len(index)))
        z = np.array([[df[c][i] for c in columns] for i in index])
        
        xticks = dict(ticks=np.arange(len(columns)), labels=np.round(columns, 2))
        yticks = dict(ticks=np.arange(len(index)), labels=np.rint(index))
        
        return x, y, z, xticks, yticks


    ### Compose your data.
    black_scholes_surface = pd.DataFrame(results[0][0],
        index= spot_space,
        columns= time_space,
    )

    ### Transform to Matplotlib friendly format.
    x, y, z, xticks, yticks = plottable_3d_info(black_scholes_surface)

    ### Set up axes and put data on the surface.
    axes = plt.figure().gca(projection='3d')
    axes.plot_wireframe(x, y, z, color='black')

    asian_surface = pd.DataFrame(results[1][0],
        index= spot_space,
        columns= time_space,
    )

    ### Transform to Matplotlib friendly format.
    x, y, z, xticks, yticks = plottable_3d_info(asian_surface)

    axes.plot_wireframe(x, y, z, color='red')


    lookback_surface = pd.DataFrame(results[2][0],
        index= spot_space,
        columns= time_space,
    )

    ### Transform to Matplotlib friendly format.
    x, y, z, xticks, yticks = plottable_3d_info(lookback_surface)

    axes.plot_wireframe(x, y, z, color='blue')



    ### Customize labels and ticks (only really necessary with
    ### non-numeric axes).
    axes.set_xlabel('time to maturity')
    axes.set_ylabel('spot')
    axes.set_zlabel('option value')
    axes.set_zlim3d(bottom=0)

    
    plt.xticks(**xticks)
    plt.yticks(**yticks)
    
    plt.locator_params(axis='x', nbins = 5, tight=True)
    plt.locator_params(axis='y', nbins = 5, tight=True)


    plt.show()
    
    
def test_monte_carlo_payoffs(spot_price, strike_price, risk_free_rate, time_horizon, asset_volatility, timesteps, num_sims, antithetic):
    spot_space = linspace(50, 150, 30)
    time_space = linspace(0.001, 5, 30)


    # generate asset paths
    asset_paths = simulate_path(spot_price, risk_free_rate, asset_volatility, time_horizon, timesteps, num_sims, antithetic)

    # Monte Carlo Asian option
    my_asian = AsianOption(asset_paths, OptionType.CALL, StrikeType.FIXED, AveragingType.GEOMETRIC, strike_price, risk_free_rate, time_horizon)
    print(f'MC Geometric Asian option call and put price: {my_asian.call_price:0.4f}, {my_asian.put_price:0.4f}')

    # Monte Carlo European Option
    my_european = EuropeanOption(asset_paths, OptionType.CALL, strike_price, risk_free_rate, time_horizon)
    print(f'MC European call and put price: {my_european.call_price:0.4f}, {my_european.put_price:0.4f}')

    # Monte Carlo Lookback option
    my_lookback = LookbackOption(asset_paths, OptionType.CALL, StrikeType.FLOATING, strike_price, risk_free_rate, time_horizon)
    print(f'MC Lookback option call and put price: {my_lookback.call_price:0.4f}, {my_lookback.put_price:0.4f}')



    results = np.zeros((3, 2, len(spot_space), len(time_space)))
    for sp_idx, sp in enumerate(spot_space, 0):
        for ts_idx, ts in enumerate(time_space, 0):
        
            asset_paths = simulate_path(sp, risk_free_rate, asset_volatility, ts, timesteps, num_sims, antithetic)    
            
            my_asian = AsianOption(asset_paths, OptionType.CALL, StrikeType.FIXED, AveragingType.GEOMETRIC, strike_price, risk_free_rate, ts)
            my_european = EuropeanOption(asset_paths, OptionType.CALL, strike_price, risk_free_rate, ts)
            my_lookback = LookbackOption(asset_paths, OptionType.CALL, StrikeType.FIXED, strike_price, risk_free_rate, ts)
                
            results[0, 0, sp_idx, ts_idx] = my_european.call_price
            results[1, 0, sp_idx, ts_idx] = my_asian.call_price
            results[2, 0, sp_idx, ts_idx] = my_lookback.call_price
            results[0, 1, sp_idx, ts_idx] = my_european.put_price
            results[1, 1, sp_idx, ts_idx] = my_asian.put_price
            results[2, 1, sp_idx, ts_idx] = my_lookback.put_price
        
        
    float_formatter = "{:.2f}".format    
    np.set_printoptions(formatter={'float_kind': float_formatter})



    def plottable_3d_info(df: pd.DataFrame):
        """
        Transform Pandas data into a format that's compatible with
        Matplotlib's surface and wireframe plotting.
        """
        index = df.index
        columns = df.columns

        x, y = np.meshgrid(np.arange(len(columns)), np.arange(len(index)))
        z = np.array([[df[c][i] for c in columns] for i in index])
        
        xticks = dict(ticks=np.arange(len(columns)), labels=np.round(columns, 2))
        yticks = dict(ticks=np.arange(len(index)), labels=np.rint(index))
        
        return x, y, z, xticks, yticks


    ### Compose your data.
    european_surface = pd.DataFrame(results[0][0],
        index= spot_space,
        columns= time_space,
    )

    ### Transform to Matplotlib friendly format.
    x, y, z, xticks, yticks = plottable_3d_info(european_surface)

    ### Set up axes and put data on the surface.
    axes = plt.figure().gca(projection='3d')
    axes.plot_wireframe(x, y, z, color='black')

    asian_surface = pd.DataFrame(results[1][0],
        index= spot_space,
        columns= time_space,
    )

    ### Transform to Matplotlib friendly format.
    x, y, z, xticks, yticks = plottable_3d_info(asian_surface)

    axes.plot_wireframe(x, y, z, color='red')


    lookback_surface = pd.DataFrame(results[2][0],
        index= spot_space,
        columns= time_space,
    )

    ### Transform to Matplotlib friendly format.
    x, y, z, xticks, yticks = plottable_3d_info(lookback_surface)

    axes.plot_wireframe(x, y, z, color='blue')



    ### Customize labels and ticks (only really necessary with
    ### non-numeric axes).
    axes.set_xlabel('time to maturity')
    axes.set_ylabel('spot')
    axes.set_zlabel('option value')
    axes.set_zlim3d(bottom=0)

    
    plt.xticks(**xticks)
    plt.yticks(**yticks)
    
    plt.locator_params(axis='x', nbins = 5, tight=True)
    plt.locator_params(axis='y', nbins = 5, tight=True)


    plt.show()
    
    
def test_asian_payoff():
    pass
