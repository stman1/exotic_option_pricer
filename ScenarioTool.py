# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 09:19:46 2022

@author: Stefan Mangold
"""

from enum import Enum
from numpy import *
from scipy.stats import gmean
from scipy.stats import norm

from Contracts import AsianOption, LookbackOption, EuropeanOption, PayoffType, OptionType, AveragingType, StrikeType 
from simulation import simulate_path

class ParameterType(Enum):
    MARKET = 1
    CONTRACT = 2
    SIMULATION = 3
    CALCULATIOMN = 4


class Parameters:
    """
    This is a class encapsulating market and contract parameters.
    
    Attributes: 

    """  
    def __init__(self, market_parameters, contract_parameters):
        # spot, strike, risk_free_rate, asset_volatility, time_to_expiry
        pass
        
        
    
class Scenario:
    
    """
    This is a class implementing a parameter scenario.
    
    Attributes: 

    """    
    
    def __init__(self, market_parameter, contract_parameter, calculation_parameter, simulation_parameter):
        
        self.market = {}
        self.contract = {}
        self.simulation = {}
        self.calculation = {}
        
        # market parameters, read in dictionaries
        for key in market_parameter:
            self.market[key] = [market_parameter[key]]
            
        # contract parameters, read in dictionaries
        for key in contract_parameter:
            self.contract[key] = contract_parameter[key]
            
        # simulation parameter, read in dictionaries
        for key in simulation_parameter:
            self.simulation[key] = simulation_parameter[key]            
            
        # market parameters, read in dictionaries
        for key in calculation_parameter:
            self.calculation[key] = calculation_parameter[key]
            
        self.scenario = {'market' : self.market, 
                         'contract' : self.contract, 
                         'simulation' : self.simulation, 
                         'calculation' : self.calculation}
    
    
    def define_scenario(self, scenario_definition):
        if scenario_definition[0] not in self.scenario:
            print(f'{scenario_definition[0]} is not a valid scenario type')
            return
            
        if scenario_definition[1] not in self.scenario[scenario_definition[0]]:
            print(f'{scenario_definition[1]} is not a valid parameter in scenario type {scenario_definition[0] }')
            return
        
        if isinstance(scenario_definition[2], (int, float)) and not isinstance(scenario_definition[2], bool):
            if isinstance(scenario_definition[3], (int, float)) and not isinstance(scenario_definition[3], bool):
                if scenario_definition[2] < scenario_definition[3]:
                    self.scenario[scenario_definition[0]][scenario_definition[1]][1].\
                        append([param_val for param_val in range(scenario_definition[2], scenario_definiton[3] + scenario_definiton[4] , scenario_definition[4])])
                    
            
        
        
        
        
    
    
    # run a scenario
    def run_scenario(self):
        self._generate_asset_paths()
       
        time_line = linspace(0, self.market['time'], self.simulation['time_steps'])
       
        if self.contract['payoff_type'] == PayoffType.EUROPEAN:
            option = EuropeanOption(time_line, 
                                     self.asset_paths, 
                                     self.contract['option_type'], 
                                     self.contract['strike'], 
                                     self.market['drift'], 
                                     self.market['time'])
           
        elif self.contract['payoff_type'] == PayoffType.ASIAN:
            option = AsianOption(time_line, 
                                 self.asset_paths, 
                                 self.contract['option_type'],
                                 self.contract['strike_type'],
                                 self.contract['averaging_type'], 
                                 self.contract['strike'], 
                                 self.market['drift'], 
                                 self.market['time'])
           
        elif self.contract['payoff_type'] == PayoffType.LOOKBACK:
            option = LookbackOption(time_line, 
                                     self.asset_paths, 
                                     self.contract['option_type'], 
                                     self.contract['strike'], 
                                     self.market['drift'], 
                                     self.market['time'])
    
        num_sims = self.simulation['num_sims']
        payoff_type = self.contract['payoff_type']
        print(f'Number of MC simulations : {num_sims}')
        print(f'{payoff_type} by Monte Carlo: {option.call_price:0.4f}, {option.put_price:0.4f}')
       
       
    def _generate_asset_paths(self):
        self.asset_paths = simulate_path(self.market['start_val'], 
                                       self.market['drift'],
                                       self.market['volatility'],
                                       self.market['time'],
                                       self.simulation['time_steps'],
                                       self.simulation['num_sims'],
                                       self.simulation['antithetic'])
               
           
       
        



  