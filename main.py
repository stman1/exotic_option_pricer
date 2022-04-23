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
from CQF_Assignment_2 import test_martingale_property_asset_price_path, test_euler_maruyama

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




#spot_space = [5, 25, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
spot_space = linspace(50, 150, 10)
#strike_space = [5, 25, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]

time_space = linspace(0.001, 5, 10)

# Closed-form Asian option
my_geometric_asian = ClosedFormGeometricAsian(spot_price, strike_price, risk_free_rate, time_horizon, asset_volatility, timesteps)
print(f'Geometric Asian option call and put price: {my_geometric_asian.call_price:0.4f}, {my_geometric_asian.put_price:0.4f}')

# Black-Scholes formula

my_black_scholes = BS(spot_price, strike_price, risk_free_rate, time_horizon, asset_volatility)
print(f'Black-Scholes option call and put price: {my_black_scholes.call_price:0.4f}, {my_black_scholes.put_price:0.4f}')

# Closed-form Lookback option
my_lookback = ClosedFormContinuousLookback(spot_price, OptionType.CALL, StrikeType.FLOATING, risk_free_rate, time_horizon, asset_volatility, strike_price, spot_price, spot_price)
print(f'Lookback option call and put price: {my_lookback.call_price:0.4f}, {my_lookback.put_price:0.4f}')

results = np.zeros((3, 2, len(spot_space), len(time_space)))
for sp_idx, sp in enumerate(spot_space, 0):
    for ts_idx, ts in enumerate(time_space, 0):
        
        my_black_scholes = BS(sp, strike_price, risk_free_rate, ts, asset_volatility)

        my_geometric_asian = ClosedFormGeometricAsian(sp, strike_price, risk_free_rate, ts, asset_volatility, timesteps)

        my_lookback = ClosedFormContinuousLookback(sp, OptionType.CALL, StrikeType.FLOATING, risk_free_rate, ts, asset_volatility, strike_price, spot_price, spot_price)

        #print(f'strike : {st} and spot : {sp}')
        #my_black_scholes.spot = sp; my_geometric_asian.spot = sp; my_lookback.spot = sp
        #my_black_scholes.strike = st; my_geometric_asian.strike = st; my_lookback.strike = st
        #my_black_scholes._price(); my_geometric_asian._price(); my_lookback._price()       
        #print(f'Black-Scholes : {my_black_scholes.call_price:0.4f}, {my_black_scholes.put_price:0.4f}')
        #print(f'Asian: {my_geometric_asian.call_price:0.4f}, {my_geometric_asian.put_price:0.4f}')
        #print(f'Lookback: {my_lookback.call_price:0.4f}, {my_lookback.put_price:0.4f}')

        results[0, 0, sp_idx, ts_idx] = my_black_scholes.call_price
        results[1, 0, sp_idx, ts_idx] = my_geometric_asian.call_price
        results[2, 0, sp_idx, ts_idx] = my_lookback.call_price
        results[0, 1, sp_idx, ts_idx] = my_black_scholes.put_price
        results[1, 1, sp_idx, ts_idx] = my_geometric_asian.put_price
        results[2, 1, sp_idx, ts_idx] = my_lookback.put_price
    
    
float_formatter = "{:.2f}".format    
np.set_printoptions(formatter={'float_kind': float_formatter})

#print(results[0][0])

grid = pd.DataFrame(results[0][0], index = spot_space, columns = time_space)


def plottable_3d_info(df: pd.DataFrame):
    """
    Transform Pandas data into a format that's compatible with
    Matplotlib's surface and wireframe plotting.
    """
    index = df.index
    columns = df.columns

    x, y = np.meshgrid(np.arange(len(columns)), np.arange(len(index)))
    z = np.array([[df[c][i] for c in columns] for i in index])
    
    xticks = dict(ticks=np.arange(len(columns)), labels=columns)
    yticks = dict(ticks=np.arange(len(index)), labels=index)
    
    return x, y, z, xticks, yticks


### Compose your data.
black_scholes_surface = pd.DataFrame(results[0][1],
    index= spot_space,
    columns= time_space,
)

### Transform to Matplotlib friendly format.
x, y, z, xticks, yticks = plottable_3d_info(black_scholes_surface)

### Set up axes and put data on the surface.
axes = plt.figure().gca(projection='3d')
axes.plot_wireframe(x, y, z, color='black')

asian_surface = pd.DataFrame(results[1][1],
    index= spot_space,
    columns= time_space,
)

### Transform to Matplotlib friendly format.
x, y, z, xticks, yticks = plottable_3d_info(asian_surface)

axes.plot_wireframe(x, y, z, color='red')


lookback_surface = pd.DataFrame(results[2][1],
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

#plt.locator_params(axis='y', nbins=5)
#plt.locator_params(axis='x', nbins=5)

plt.show()



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
