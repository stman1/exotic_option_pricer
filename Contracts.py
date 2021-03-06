# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:30:09 2022

Classes to define contracts and their payoffs for use in Monte Carlo pricing

@author: Stefan Mangold
"""
from enum import Enum
from numpy import *
from scipy.stats import gmean

class OptionType(Enum):
    CALL = 1
    PUT = 2
        
class StrikeType(Enum):
    FIXED = 1
    FLOATING = 2
    
class AveragingType(Enum):
    ARITHMETIC = 1
    GEOMETRIC = 2
    
    
class PayoffType(Enum):
    EUROPEAN = 1
    ASIAN = 2
    LOOKBACK = 3


class EuropeanOption:
    """
    This is a class capturing European plain vanilla option contracts.
    
    Attributes: 
        time_line               : np.array
        asset_paths             : np.array(rows: num_simulations, columns: num_observations)
        option_type             : Enum type (Call / Put)
        strike                  : int or float 
        tte                     : int or float [time to expiry, expressed as year fraction]
    """ 
    def __init__(self, asset_paths, option_type, strike, rate, tte):
        # Array of asset paths
        self.s_mat = asset_paths
        
        # Option type, call or put
        self.opt_type = option_type
               
        # Strike, in case it is of type fixed strike
        self.strike = strike
        
        # risk-free interest rate per annum
        self.rate = rate
        
        # time to expiration (year fraction)
        self.tte = tte
        

        # The __dict__ attribute
        '''
        Contains all the attributes defined for the object itself. It maps the attribute name to its value.
        '''
        for i in ['callPrice', 'putPrice', 'callDelta', 'putDelta', 'callTheta', 'putTheta', \
                  'callRho', 'putRho', 'vega', 'gamma']:
            self.__dict__[i] = None
        
        [self.call_price, self.put_price] = self._price()

    def _price(self):
        discount_factor = exp(-self.rate * self.tte)
        call_price = discount_factor * mean(maximum(self.s_mat[-1] - self.strike, 0.))
        put_price = discount_factor * mean(maximum(self.strike - self.s_mat[-1], 0.))
        return [call_price, put_price]


class AsianOption:    
    """
    This is a class capturing Asian Option contracts.
    
    Attributes: 
        time_line       : np.array
        asset_paths     : np.array(rows: num_simulations, columns: num_observations)
        option_type     : Enum type (Call / Put)
        strike_type     : Enum type (Fixed Strike / Floating Strike)
        averaging_type  : Enum type (Arithmetic / Geometric average)
        strike          : int or float 
        tte             : int or float [time to expiry, expressed as year fraction]
    """ 
    
    def __init__(self, asset_paths, option_type, strike_type, averaging_type, strike, rate, tte):        
        # Array of asset paths
        self.s_mat = asset_paths
        
        # Option type, call or put
        self.opt_type = option_type
        
        # Strike type, fixed strike or floating strike
        self.strike_type = strike_type
        
        # type of averaging, arithmetic or geometric averaging
        self.avg_type = averaging_type
       
        # Strike, in case it is of type fixed strike
        self.strike = strike
        
        # risk-free interest rate per annum
        self.rate = rate
        
        # time to expiration (year fraction)
        self.tte = tte
        
        if self.avg_type == AveragingType.ARITHMETIC:
            self.avg = mean(self.s_mat, axis = 0)
        elif self.avg_type == AveragingType.GEOMETRIC:
            self.avg = gmean(self.s_mat, axis = 0)
        else:
            raise TypeError("The averaging type must be either arithmetic or geometric.")
            
    
        # The __dict__ attribute
        '''
        Contains all the attributes defined for the object itself. It maps the attribute name to its value.
        '''
        for i in ['callPrice', 'putPrice']:
            self.__dict__[i] = None
        
        [self.call_price, self.put_price] = self._price()

    def _price(self):
        
        discount_factor = exp(-self.rate * self.tte)
        if self.strike_type == StrikeType.FIXED:
            call_price = discount_factor * mean(maximum(self.avg - self.strike, 0.))
            put_price = discount_factor * mean(maximum(self.strike - self.avg, 0.))
        elif self.strike_type  == StrikeType.FLOATING:
            call_price = discount_factor * mean(maximum(self.s_mat[-1] - self.avg, 0.))
            put_price = discount_factor * mean(maximum(self.avg - self.s_mat[-1], 0.))
        else:
            raise TypeError(f'{self.strike_type}: strike type must be either FIXED or FLOATING.')
                    
        return [call_price, put_price]
            
    
    def resetAveragingType(self, averaging_type):
        # reset averaging type
        if averaging_type == AveragingType.ARITHMETIC:
            self.avg = mean(self.s_mat, axis = 0)
        elif averaging_type == AveragingType.GEOMETRIC:
            self.avg = gmean(self.s_mat, axis = 0)
        else:
            raise TypeError("The averaging type must be either arithmetic or geometric.")
        # update price
        [self.call_price, self.put_price] = self._price()
            
    
class LookbackOption:    
    """
    This is a class capturing Lookback option contracts.
    
    Attributes: 
        time_line               : np.array
        asset_paths             : np.array(rows: num_simulations, columns: num_observations)
        option_type             : Enum type (Call / Put)
        strike                  : int or float 
        tte                     : int or float [time to expiry, expressed as year fraction]
    """ 
    
    def __init__(self, asset_paths, option_type, strike_type, strike, rate, tte):
        # Array of asset paths
        self.s_mat = asset_paths
        
        # Option type, call or put
        self.opt_type = option_type
        
        # Strike type, fixed strike or floating strike
        self.st_type = strike_type
               
        # Strike, in case it is of type fixed strike
        self.strike = strike
        
        # risk-free interest rate per annum
        self.rate = rate
        
        # time to expiration (year fraction)
        
        self.tte = tte
        

            
    
        # The __dict__ attribute
        '''
        Contains all the attributes defined for the object itself. It maps the attribute name to its value.
        '''
        for i in ['call_price', 'put_price']:
            self.__dict__[i] = None
        
        [self.call_price, self.put_price] = self._price()

    def _price(self):
        discount_factor = exp(-self.rate * self.tte)    
        if self.st_type == StrikeType.FLOATING:
            call_price = discount_factor * mean(self.s_mat[-1] - amin(self.s_mat, axis = 0))
            put_price = discount_factor *mean(amax(self.s_mat, axis = 0) - self.s_mat[-1])
        elif  self.st_type == StrikeType.FIXED:
            call_price = discount_factor * mean(maximum(amax(self.s_mat, axis = 0) - self.strike, 0.))
            put_price = discount_factor * mean(maximum(self.strike - amin(self.s_mat, axis = 0), 0.))
        else:
            call_price = 0; put_price = 0
        return [call_price, put_price]
            
    
       
    