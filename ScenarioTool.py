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
    CALCULATION = 4


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
        
        # check market scenarios
        market_scenarios = self._find_scenario('market')
        for ms in market_scenarios:
            self._run_market_scenario(ms)
        
        # check contract scenarios
        # if contract parameters are present, run them
        contract_scenarios = self._find_scenario('contract')
        for cs in contract_scenarios:
            self._run_contract_scenario(cs)
        
        # check calculation scenario
        # if calculation parameters are present, run them
        # self._run_calculation_scenario()


        # check simulation scenario
        simulation_scenarios = self._find_scenario('simulation')
        
        for ss in simulation_scenarios:
            self._run_simulation_scenario(ss)  
        
    
    
    # run a scenario
    def _run_single_scenario(self):
        option = self._instantiate_contract()
        num_sims = self.simulation['num_sims']['eval']
        payoff_type = self.contract['payoff_type']['eval']
        return [option.call_price, option.put_price]
       
       
    def _instantiate_contract(self):
        if self.contract['payoff_type']['eval'] == PayoffType.EUROPEAN:
            option = EuropeanOption(self.asset_paths, 
                                     self.contract['option_type']['eval'], 
                                     self.contract['strike']['eval'], 
                                     self.market['drift']['eval'], 
                                     self.market['time']['eval'])
           
        elif self.contract['payoff_type']['eval'] == PayoffType.ASIAN:
            option = AsianOption(self.asset_paths, 
                                 self.contract['option_type']['eval'],
                                 self.contract['strike_type']['eval'],
                                 self.contract['averaging_type']['eval'], 
                                 self.contract['strike']['eval'], 
                                 self.market['drift']['eval'], 
                                 self.market['time']['eval'])
           
        elif self.contract['payoff_type']['eval'] == PayoffType.LOOKBACK:
            option = LookbackOption(self.asset_paths, 
                                     self.contract['option_type']['eval'], 
                                     self.contract['strike']['eval'], 
                                     self.market['drift']['eval'], 
                                     self.market['time']['eval'])
        else:
            payoff_type_input = self.contract['payoff_type']['eval']
            print(f'Contract was not instantiated: contract type {payoff_type_input} unknown.')
        
        return option
            
        
           
        
        
    def _run_market_scenario(self, market_scenario):
        # market scenarios always need a regeneration of asset paths
        print(". . . running market scenarios.")
        try:
            # save original parameter value
            original_parameter_val = self.market[market_scenario[0]]['eval']
            # unwrap scenario
            market_parameter_space = self.market[market_scenario[0]]['scenario']

            scenario_call = []; scenario_put = []
            for param in market_parameter_space:
                # generate asset paths
                self._generate_asset_paths()
                # change parameter in scenario
                self.market[market_scenario[0]]['eval'] = param
                [call, put] = self._run_single_scenario()
                scenario_call.append(call)
                scenario_put.append(put)
                
            # reset state of the object    
            self.market[market_scenario[0]]['eval'] = original_parameter_val
        except:
            # reset state of the object    
            self.market[market_scenario[0]]['eval'] = original_parameter_val
            market_parameter_space = 'null'
            
        return scenario_call, scenario_put
        
    
    def _run_simulation_scenario(self, simulation_scenario):
        # simulation scenarios always need a regeneration of asset paths
        print(". . . running simulation scenarios.")
        try:
            # save original parameter value
            original_parameter_val = self.simulation[simulation_scenario[0]]['eval']
            # unwrap scenario
            simulation_parameter_space = self.simulation[simulation_scenario[0]]['scenario']

            scenario_call = []; scenario_put = []
            for param in simulation_parameter_space:
                # generate asset paths
                self._generate_asset_paths()
                # change parameter in scenario
                self.simulation[simulation_scenario[0]]['eval'] = param
                [call, put] = self._run_single_scenario()
                scenario_call.append(call)
                scenario_put.append(put)
                
            # reset state of the object    
            self.simulation[simulation_scenario[0]]['eval'] = original_parameter_val
        except:
            # reset state of the object    
            self.simulation[simulation_scenario[0]]['eval'] = original_parameter_val
            simulation_parameter_space = 'null'
            
        return scenario_call, scenario_put
    
    
    def _run_contract_scenario(self, contract_scenario):
        # a contract does not need newly generated asset paths
        print(". . . running contract scenarios.")
        try:
            # save original parameter value
            original_parameter_val = self.contract[contract_scenario[0]]['eval']
            # unwrap scenario
            contract_parameter_space = self.contract[contract_scenario[0]]['scenario']
            # generate asset paths
            self._generate_asset_paths()
            # instantiate contract
            scenario_call = []; scenario_put = []
            for param in contract_parameter_space:
                # change parameter in scenario
                self.contract[contract_scenario[0]]['eval'] = param
                [call, put] = self._run_single_scenario()
                scenario_call.append(call)
                scenario_put.append(put)
                
            # reset state of the object    
            self.contract[contract_scenario[0]]['eval'] = original_parameter_val
        except:
            # reset state of the object    
            self.contract[contract_scenario[0]]['eval'] = original_parameter_val
            contract_parameter_space = 'null'
            
        return scenario_call, scenario_put
        
    
    def _run_calculation_scenario(self):
        # calculation scenarios represent a comparison of one calculation method with another
        pass
    
    
    def _find_scenario(self, parameter_type):
        scenario_parameter = []
        for first in self.scenario[parameter_type].keys():
            if 'scenario' in list(self.scenario[parameter_type][first].keys()):
                scenario_parameter.append([first, list(self.scenario[parameter_type][first]['scenario'])])
        return scenario_parameter

        
    
    def _generate_asset_paths(self):
        self.asset_paths = simulate_path(self.market['start_val']['eval'], 
                                       self.market['drift']['eval'],
                                       self.market['volatility']['eval'],
                                       self.market['time']['eval'],
                                       self.simulation['time_steps']['eval'],
                                       self.simulation['num_sims']['eval'],
                                       self.simulation['antithetic']['eval'])
    
    
    

        
             
       
        



  