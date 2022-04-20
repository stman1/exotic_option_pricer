# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing 
import time
import pandas as pd
from numpy import *
from simulation import simulate_path, moment_matching, simulate_path_moment_matched
from BlackScholes import BS
from Contracts import AsianOption, LookbackOption, EuropeanOption, PayoffType, OptionType, AveragingType, StrikeType
from ScenarioTool import Scenario
from CQF_Assignment_2 import test_martingale_property_asset_price_path

import matplotlib
import matplotlib.pyplot as plt

# Plot settings
plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['lines.linewidth'] = 2.0

# Set max row to 300
pd.set_option('display.max_rows', 300)


# Global initial market, contract and simulation parameters

spot_price = 100; strike_price = 100; risk_free_rate = 0.05; asset_volatility = 0.20; time_horizon = 1; timesteps = 252; antithetic = True

# Test variance reduction technique: moment-matching 
#start_time = time.time()
#phi = moment_matching(0., 1., 1000, 5)
#asset_paths = simulate_path_moment_matched(spot_price, risk_free_rate, asset_volatility, time_horizon, timesteps, 100)
#time_step_mean = mean(phi, axis = 0)
#time_step_std = std(phi, axis = 0)
#print(shape(phi))
#print("--- %s seconds ---" % (time.time() - start_time))

# Test asset price path convergence
#start_time = time.time()
#asset_price_path_convergence = test_martingale_property_asset_price_path(spot_price, risk_free_rate, asset_volatility, timesteps, antithetic)
#print("--- %s seconds ---" % (time.time() - start_time))


# Validate European option payoff against analytical Black Scholes pricer

# =============================================================================
# num_sims = 1000000
# 
# start_time = time.time()
# asset_paths = simulate_path(spot_price, risk_free_rate, asset_volatility, time_horizon, timesteps, num_sims, True)
# time_line = linspace(0, time_horizon, timesteps)
# option_type = OptionType.CALL
# european = EuropeanOption(time_line, asset_paths, option_type, strike_price, risk_free_rate, time_horizon)
# print(f'Number of MC simulations : {num_sims}')
# print(f'European Option by Monte Carlo: {european.call_price:0.4f}, {european.put_price:0.4f}')
# print("--- %s seconds ---" % (time.time() - start_time))
# 
# # Setup BS analytic pricer
# 
# analytic_bs = BS(100, 100, 0.05, 1, 0.2)
# print(f'Analytic BS Call Option price: {analytic_bs.callPrice:0.4f}')
# print(f'Analytic BS Put Option price: {analytic_bs.putPrice:0.4f}')


# Run a scenario

market_parameters = {'start_val': 100, 'drift' : 0.05, 'volatility' : 0.2, 'time' : 1}
contract_parameters = { 'payoff_type' : PayoffType.EUROPEAN,  'option_type' : OptionType.CALL, 'strike' : 100}
calculation_parameters = { 'analytic' : True, 'monte_carlo' : False}
simulation_parameters = {'num_sims': 10000, 'time_steps' : 252, 'antithetic' : True}

my_scenario = Scenario(market_parameters, contract_parameters, calculation_parameters, simulation_parameters)

my_scenario.run_scenario()
# 
# =============================================================================

# =============================================================================
# # Call the simulation function
# S = simulate_path(100, 0.05, 0.20, 1, 252, 100000)
# 
# # Define parameters
# K = 100.; r = 0.05; T = 1
# 
# # European options
# 
# # Calculate the discounted value of the expeced payoff
# C0 = exp(-r*T) * mean(maximum(S[-1] - K, 0))
# P0 = exp(-r*T) * mean(maximum(K - S[-1], 0))
# 
# # Print the values
# print(f'European Call Option Value: {C0:0.4f}')
# print(f'European Put Option Value: {P0:0.4f}')
# 
# # Asian options
# 
# # Average price
# A = S.mean(axis=0)   
#     
# # Calculate the discounted value of the expeced payoff
# C0 = exp(-r*T) * mean(maximum(A - K, 0))                # mean is for present value which is for discounting
# P0 = exp(-r*T) * mean(maximum(K - A, 0))
# 
# # Print the values
# print(f'Asian Call Option Value: {C0:0.4f}')
# print(f'Asian Put Option Value: {P0:0.4f}')
# 
# # Black-Scholes price
# 
# bsopt = BS(100, 100, 0.05, 1, 0.2)
# 
# header = ['Option Price', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho']
# table = [[bsopt.callPrice, bsopt.callDelta, bsopt.gamma, bsopt.callTheta, bsopt.vega, bsopt.callRho]]
# 
# print(f'Black-Scholes Call Option Value: {bsopt.callPrice:0.4f}')
# print(f'Black-Scholes Put Option Value: {bsopt.putPrice:0.4f}')
# 
# # Define Asian option payoffs
# 
# time_line = linspace(0, T, 252)
# asianOption = AsianOption(time_line, S, OptionType.CALL, StrikeType.FIXED, AveragingType.ARITHMETIC, 100, r, T)
# #print(asianOption.call_price)
# print(f'Arithmetic Asian Option by Monte Carlo: {asianOption.call_price:0.4f}, {asianOption.put_price:0.4f}')
# asianOption.resetAveragingType(AveragingType.GEOMETRIC)
# print(f'Geometric Asian Option by Monte Carlo: {asianOption.call_price:0.4f}, {asianOption.put_price:0.4f}')
# 
# =============================================================================
