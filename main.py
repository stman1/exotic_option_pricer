# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Importing 
import time
import pandas as pd
import numpy as np
from numpy import *
from simulation import simulate_path, moment_matching, simulate_path_moment_matched
from BlackScholes import BS
from AnalyticGeometricAsian import ClosedFormGeometricAsian
from AnalyticContinuousLookback import ClosedFormContinuousLookback
from Contracts import AsianOption, LookbackOption, EuropeanOption, PayoffType, OptionType, AveragingType, StrikeType
from ScenarioTool import Scenario
from CQF_Assignment_2 import test_martingale_property_asset_price_path, test_martingale_property_asset_price_path_repeated, test_euler_maruyama, test_closed_form_solutions, test_monte_carlo_payoffs

import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plot settings
plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['lines.linewidth'] = 2.0

# Set max row to 300
pd.set_option('display.max_rows', 300)


# Global initial market, contract and simulation parameters


# market parameters
spot_price = 100 
risk_free_rate = 0.05 
asset_volatility = 0.2
time_horizon = 1 

# contract parameters
strike_price = 100 

#simulation parameters
num_simulations = 100000
timesteps = 252 
antithetic_flag = True


#asset_paths = simulate_path(spot_price, risk_free_rate, asset_volatility, time_horizon, timesteps, num_simulations, antithetic_flag)

#my_lookback = LookbackOption(asset_paths, OptionType.CALL, StrikeType.FLOATING, strike_price, risk_free_rate, time_horizon)
#print(f'MC Lookback option call and put price: {my_lookback.call_price:0.4f}, {my_lookback.put_price:0.4f}')

#res = test_martingale_property_asset_price_path_repeated(spot_price, risk_free_rate, asset_volatility, timesteps, antithetic_flag, 10)

test_closed_form_solutions(spot_price, strike_price, risk_free_rate, time_horizon, asset_volatility, timesteps)

#test_monte_carlo_payoffs(spot_price, strike_price, risk_free_rate, time_horizon, asset_volatility, timesteps, num_simulations, antithetic_flag)

#ig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
##ax = Axes3D(fig)
#surf = ax.plot_wireframe(grid.index, grid.columns, grid)
#plt.show()


# figure, axes = plt.subplots(1, 2, figsize = (20,6), sharey=True)
# 
# x = df['Time Steps']
# 
# d = [df['Call price difference'], df['Put price difference']]
# 
# 
# axes[0].plot(x, d[0], label='Call price diff')
# axes[1].plot(x, d[1], label='Put price diff')
# 
# # Set axis title
# axes[0].set_title('Call Price'), axes[1].set_title('Put Price')
# 
# # Define legend
# axes[0].legend(), axes[1].legend()
# 
# plt.show()





# =============================================================================
# start_time = time.time()
# df = test_euler_maruyama(100, 100, 0.05, 0.2, 1, 252, False, False, True)
# print("--- %s seconds ---" % (time.time() - start_time))
# 
# figure, axes = plt.subplots(1, 2, figsize = (20,6), sharey=True)
# 
# x = df['Time Steps']
# 
# d = [df['Call price difference'], df['Put price difference']]
# 
# 
# axes[0].plot(x, d[0], label='Call price diff')
# axes[1].plot(x, d[1], label='Put price diff')
# 
# # Set axis title
# axes[0].set_title('Call Price'), axes[1].set_title('Put Price')
# 
# # Define legend
# axes[0].legend(), axes[1].legend()
# 
# plt.show()
# =============================================================================

# Run a scenario

# =============================================================================
# market_parameters = {'start_val': spot_price, 'drift' : risk_free_rate, 'volatility' : asset_volatility, 'time' : time_horizon}
# contract_parameters = { 'payoff_type' : PayoffType.ASIAN,  'strike_type' : StrikeType.FIXED, 'averaging_type' : AveragingType.ARITHMETIC, 'option_type' : OptionType.CALL, 'strike' : 100}
# calculation_parameters = { 'analytic' : True, 'monte_carlo' : False}
# simulation_parameters = {'num_sims': num_simulations, 'time_steps' : timesteps, 'antithetic' : antithetic_flag}
# 
# my_scenario = Scenario(market_parameters, contract_parameters, calculation_parameters, simulation_parameters)
# 
# 
# 
# strike_space = [0, 25, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
# my_scenario.define_scenario(['contract', 'strike', strike_space])
# averaging_space = [AveragingType.ARITHMETIC, AveragingType.GEOMETRIC]
# my_scenario.define_scenario(['contract', 'averaging_type', averaging_space])
# 
# 
# spot_space = [0, 25, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
# my_scenario.define_scenario(['market', 'start_val', spot_space])
# 
# 
# simulation_space = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
# my_scenario.define_scenario(['simulation', 'num_sims', simulation_space])
# 
# 
# my_scenario.run_scenarios()
# 
# =============================================================================
