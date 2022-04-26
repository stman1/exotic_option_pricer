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
visualize_payoff)


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
num_simulations = 10000
timesteps = 252 
antithetic_flag = True

# Vol = True, time = False
vol_or_time_flag = False

# Call = True, Put = False

option_type = OptionType.CALL
strike_type = StrikeType.FLOATING
averaging_type = AveragingType.GEOMETRIC


opt_price_array = []
call_rmse_array = np.zeros((4, 4))
put_rmse_array = np.zeros((4, 4))

for n_idx, n in enumerate([1000, 10000, 100000, 1000000]):
    for ts_idx, ts in enumerate([252, 504, 756, 1008]):
        option_prices, call_rmse, put_rmse, x_var, y_var = test_european_payoff(option_type, 
                                                                                strike_price, 
                                                                                risk_free_rate, 
                                                                                asset_volatility, 
                                                                                ts, 
                                                                                n, 
                                                                                antithetic_flag)
        opt_price_array.append(option_prices)
        call_rmse_array[n_idx][ts_idx] = call_rmse
        put_rmse_array[n_idx][ts_idx] = put_rmse



