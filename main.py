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
from CQF_Assignment_2 import (test_martingale_property_asset_price_path, 
test_martingale_property_asset_price_path_repeated, 
test_euler_maruyama, 
test_closed_form_solutions, 
test_monte_carlo_payoffs,
test_lookback_payoff,
test_asian_payoff,
test_european_payoff, 
visualize_line_plot,
visualize_grid_plot,
root_mean_squared_error)


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
num_simulations = 1000000
timesteps = 2016 
antithetic_flag = True

# Vol = True, time = False
vol_or_time_flag = False

# Call = True, Put = False

option_type = OptionType.CALL
strike_type = StrikeType.FIXED
averaging_type = AveragingType.GEOMETRIC

spot_space = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
time_space = [1, 0.5, 0.25, 0.166, 0.0833, 0.027777, 0.00396825]

mc_option_prices = test_monte_carlo_payoffs(spot_space,
                                          time_space,
                                          option_type,
                                          strike_type,
                                          averaging_type,
                                          spot_price, 
                                          strike_price, 
                                          risk_free_rate,
                                          time_horizon, 
                                          asset_volatility, 
                                          timesteps, 
                                          num_simulations, 
                                          antithetic_flag)

#visualize_grid_plot(mc_option_prices, spot_space, time_space)


cf_option_prices = test_closed_form_solutions(spot_space,
                                              time_space,
                                              option_type,
                                              strike_type,
                                              averaging_type,
                                              spot_price, 
                                              strike_price, 
                                              risk_free_rate, 
                                              time_horizon, 
                                              asset_volatility, 
                                              timesteps)


#visualize_grid_plot(cf_option_prices, spot_space, time_space)

mc_asian_prices = mc_option_prices[1]; cf_asian_prices = cf_option_prices[1]
mc_lookback_prices = mc_option_prices[2]; cf_lookback_prices = cf_option_prices[2]

call_rmse_asian, put_rmse_asian = root_mean_squared_error(mc_asian_prices, cf_asian_prices)
call_rmse_lookback, put_rmse_lookback = root_mean_squared_error(mc_lookback_prices, cf_lookback_prices)  

print(f'Asian option rmse call:   {call_rmse_asian:0.4f}, put:   {put_rmse_asian:0.4f}')
print(f'Lookback option rmse call:   {call_rmse_lookback:0.4f}, put:   {put_rmse_lookback:0.4f}')


# opt_price_array = []
# call_rmse_array = np.zeros((4, 4))
# put_rmse_array = np.zeros((4, 4))

# for n_idx, n in enumerate([1000, 10000, 100000, 1000000]):
#     for ts_idx, ts in enumerate([252, 504, 756, 1008]):
#         option_prices, call_rmse, put_rmse, x_var, y_var = test_european_payoff(option_type, 
#                                                                                 strike_price, 
#                                                                                 risk_free_rate, 
#                                                                                 asset_volatility, 
#                                                                                 ts, 
#                                                                                 n, 
#                                                                                 antithetic_flag)
#         opt_price_array.append(option_prices)
#         call_rmse_array[n_idx][ts_idx] = call_rmse
#         put_rmse_array[n_idx][ts_idx] = put_rmse



