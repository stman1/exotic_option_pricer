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


# market parameters
spot_price = 100 
risk_free_rate = 0.05 
asset_volatility = 0.20 
time_horizon = 1 

# contract parameters
strike_price = 100 

#simulation parameters
num_simulations = 10000
timesteps = 252 
antithetic_flag = True

# Run a scenario

market_parameters = {'start_val': spot_price, 'drift' : risk_free_rate, 'volatility' : asset_volatility, 'time' : time_horizon}
contract_parameters = { 'payoff_type' : PayoffType.ASIAN,  'strike_type' : StrikeType.FIXED, 'averaging_type' : AveragingType.ARITHMETIC, 'option_type' : OptionType.CALL, 'strike' : 100}
calculation_parameters = { 'analytic' : True, 'monte_carlo' : False}
simulation_parameters = {'num_sims': num_simulations, 'time_steps' : timesteps, 'antithetic' : antithetic_flag}

my_scenario = Scenario(market_parameters, contract_parameters, calculation_parameters, simulation_parameters)



strike_space = [0, 25, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
my_scenario.define_scenario(['contract', 'strike', strike_space])
averaging_space = [AveragingType.ARITHMETIC, AveragingType.GEOMETRIC]
my_scenario.define_scenario(['contract', 'averaging_type', averaging_space])


spot_space = [0, 25, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
my_scenario.define_scenario(['market', 'start_val', spot_space])


simulation_space = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
my_scenario.define_scenario(['simulation', 'num_sims', simulation_space])


my_scenario.run_scenarios()

