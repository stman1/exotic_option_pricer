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
            self.market[key] = {'eval': market_parameter[key]}
            
        # contract parameters, read in dictionaries
        for key in contract_parameter:
            self.contract[key] = {'eval': contract_parameter[key]}
            
        # simulation parameter, read in dictionaries
        for key in simulation_parameter:
            self.simulation[key] = {'eval': simulation_parameter[key]}
            
        # market parameters, read in dictionaries
        for key in calculation_parameter:
            self.calculation[key] = {'eval': calculation_parameter[key]}
            
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
        self.scenario[scenario_definition[0]][scenario_definition[1]]['scenario'] =  scenario_definition[2]
            
        
        
    def run_scenarios(self):
        
        # check each of market, contract, calculation, and simulation parameters for scenarios
        
        # check market scenario
        # if market parameters are present, run them
        # self._run_market_scenario()
        
        # check contract scenario
        # if contract parameters are present, run them
        contract_scenarios = self._find_scenario('contract')
        
        for cs in contract_scenarios:
            self._run_contract_scenario(cs)
        
        # check calculation scenario
        # if calculation parameters are present, run them
        # self._run_calculation_scenario()


        # check simulation scenario
        # if simulation parameters are present, run them
        # self._run_simulation_scenario()        
        pass
        
    
    
    # run a scenario
    def _run_single_scenario(self):
        self._generate_asset_paths()
       
        time_line = linspace(0, self.market['time']['eval'], self.simulation['time_steps']['eval'])
       
        option = self._instantiate_contract(time_line)
    
        num_sims = self.simulation['num_sims']['eval']
        payoff_type = self.contract['payoff_type']['eval']
        print(f'Number of MC simulations : {num_sims}')
        print(f'{payoff_type} by Monte Carlo: {option.call_price:0.4f}, {option.put_price:0.4f}')
       
       
    def _instantiate_contract(self, time_line):
        if self.contract['payoff_type']['eval'] == PayoffType.EUROPEAN:
            option = EuropeanOption(time_line, 
                                     self.asset_paths, 
                                     self.contract['option_type']['eval'], 
                                     self.contract['strike']['eval'], 
                                     self.market['drift']['eval'], 
                                     self.market['time']['eval'])
           
        elif self.contract['payoff_type']['eval'] == PayoffType.ASIAN:
            option = AsianOption(time_line, 
                                 self.asset_paths, 
                                 self.contract['option_type']['eval'],
                                 self.contract['strike_type']['eval'],
                                 self.contract['averaging_type']['eval'], 
                                 self.contract['strike']['eval'], 
                                 self.market['drift']['eval'], 
                                 self.market['time']['eval'])
           
        elif self.contract['payoff_type']['eval'] == PayoffType.LOOKBACK:
            option = LookbackOption(time_line, 
                                     self.asset_paths, 
                                     self.contract['option_type']['eval'], 
                                     self.contract['strike']['eval'], 
                                     self.market['drift']['eval'], 
                                     self.market['time']['eval'])
        else:
            payoff_type_input = self.contract['payoff_type']['eval']
            print(f'Contract was not instantiated: contract type {payoff_type_input} unknown.')
        
        return option
            
        
        
        
        
        
    def _run_market_scenario(self):
        # market scenarios always need a regeneration of asset paths
        pass
    
    def _run_simulation_scenario(self):
        #  simulation scenarios also always need a regeneration of asset paths
        pass
    
    
    def _run_contract_scenario(self, contract_scenario):
        # a contract does not need newly generated asset paths
        
        try:
            original_parameter_val = self.contract[contract_scenario[0]]['eval']
            # unwrap scenario
            contract_parameter_space = self.contract[contract_scenario[0]]['scenario']
            # instantiate contract
            
            for param in contract_parameter_space:
                # change parameter in scenario
                self.contract[contract_scenario[0]]['eval'] = param
                self._run_single_scenario()
                
            # reset state of the object    
            self.contract[contract_scenario[0]]['eval'] = original_parameter_val
        except IndexError:
            # reset state of the object    
            self.contract[contract_scenario[0]]['eval'] = original_parameter_val
            contract_parameter_space = 'null'
        
    
    def _run_calculation_scenario(self):
        # calculation scenarios represent a comparison of one calculation method with another
        pass
    
    
    def _find_scenario(self, parameter_type):
        scenario_parameter = []
        for first in self.scenario['contract'].keys():
            if 'scenario' in list(self.scenario['contract'][first].keys()):
                scenario_parameter.append([first, list(self.scenario['contract'][first]['scenario'])])
        return scenario_parameter

        
    
    def _generate_asset_paths(self):
        self.asset_paths = simulate_path(self.market['start_val']['eval'], 
                                       self.market['drift']['eval'],
                                       self.market['volatility']['eval'],
                                       self.market['time']['eval'],
                                       self.simulation['time_steps']['eval'],
                                       self.simulation['num_sims']['eval'],
                                       self.simulation['antithetic']['eval'])
    
    
    

        
             
       
        



  