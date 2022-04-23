# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 13:07:28 2022

@author: Stefan Mangold
"""

from numpy import *
from string import Template
from scipy.stats import norm
from Contracts import StrikeType, OptionType

class ClosedFormContinuousLookback:
    """
    This is a class for Options contract for pricing European options on stocks without dividends.
    
    Attributes: 
        spot          : int or float
        strike_type   : Enum type
        rate          : float
        dte           : int or float [days to expiration in number of years]
        volatility    : float
        strike        : int or float
        realized_max   : int or float
        realized_min   : int or float
    """    
    
    def __init__(self, spot, option_type, strike_type, rate, dte, volatility, strike, realized_max, realized_min):
        
        # Spot Price
        self.spot = spot
        
        # largest asset value observed since contract outset
        self.realized_max = realized_max
        
        # smallest asset value observed since contract outset
        self.realized_min = realized_min
        
        # Option type, call or put
        self.opt_type = option_type
        
        # Strike type, fixed or floating strike
        self.st_type = strike_type
        
        # Option Strike
        self.strike = strike
        
        # Interest Rate
        self.rate = rate
        
        # Days To Expiration
        self.dte = dte
        
        # Volatility
        self.volatility = volatility
        
        [self.call_price, self.put_price] = self._price()
               

    def _assign_pricing_formula_inputs(self):
        if self.st_type == StrikeType.FLOATING:
            
            if self.opt_type == OptionType.CALL:
                values = ['+', '+', '-', 'self.realized_min', '+', '+', '-', '+','-', '']    
            elif self.opt_type == OptionType.PUT:
                values = ['-', '-', '+', 'self.realized_max', '-', '-', '+', '-','+', '']              
            else:
                values = None
                # raise an exception
                raise TypeError(f'Lookback option type is {self.opt_type} but must be either OptionType.CALL or OptionType.PUT') 
                      
        elif self.st_type == StrikeType.FIXED:   
            if self.opt_type == OptionType.CALL:
                if self.strike >= self.realized_max:
                    values = ['+', '+', '-', 'self.strike', '+', '-', '+', '-','+', '']
                else:
                    values = ['+', '+', '-', 'self.realized_max', '+', '-', '+', '-','+', '(self.realized_max - self.strike) * e**(-self.rate*self.dte)']  
                
            elif self.opt_type == OptionType.PUT:
                if self.strike <= self.realized_min: 
                    values = ['-', '-', '+', 'self.strike', '-', '+', '-', '+','-', '']
                else:
                    values = ['-', '-', '+', 'self.realized_min', '-', '+', '-', '+','-', '(self.strike - self.realized_min) * e**(-self.rate*self.dte)']  
            else:
                values = None
                # raise an exception
                raise TypeError(f'Lookback option type is {self.opt_type} but must be either OptionType.CALL or OptionType.PUT') 

        else:
            values = None
            # raise an exception
            raise TypeError(f'Lookback option type is {self.st_type} but must be either StrikeType.FLOATING or StrikeType.FIXED') 
        
        return values
        
    
    
    def _create_formula_input_dictionary(self):
        
        keys = ['sign_spot_1st',
        'sign_d1_1st',
        'sign_strike_2nd',
        'min_max_fixed_as_strike',
        'sign_d2_2nd',
        'sign_3rd',
        'sign_d1_3rd',
        'sign_correction_3rd',
        'sign_last',
        'additional_term']
        
        
        values = self._assign_pricing_formula_inputs()
        if values == None:
            print("ERROR: input values to templated pricing formulae are empty. ")
        
        template_inputs = dict(zip(keys, values))
        
        return template_inputs
        
   
    def _template_pricing_formula(self):
        
        template_input = self._create_formula_input_dictionary()
        
        # evaluate d1, d2 first
        d1_template = Template('(log(self.spot / $strike) + (self.rate + 0.5 * self.volatility**2)*self.dte)\
                             / (self.volatility * sqrt(self.dte))')
                             
        d1_eval = d1_template.substitute(strike = template_input['min_max_fixed_as_strike'])
        self._d1_ = eval(d1_eval)
        self._d2_ = self._d1_ - self.volatility * sqrt(self.dte) 
                
        price_formula_template = Template('$additional_term $sign_spot_1st self.spot * norm.cdf($sign_d1_1st self._d1_)\
                                          $sign_strike_2nd $strike * e**(-self.rate * self.dte)\
                                          * norm.cdf($sign_d2_2nd self._d2_) +self.spot * e**(-self.rate * self.dte)\
                                          * self.volatility**2 / (2*(self.rate)) * ($sign_3rd (self.spot/$strike)**(-(2*self.rate/(self.volatility**2)))\
                                          * norm.cdf($sign_d1_3rd self._d1_ $sign_correction_3rd (2*self.rate*sqrt(self.dte))/(self.volatility))\
                                          $sign_last e**(self.rate*self.dte) * norm.cdf(-self._d1_)) ')

        option = price_formula_template.substitute(sign_spot_1st =  template_input['sign_spot_1st'], 
                                                sign_d1_1st = template_input['sign_d1_1st'], 
                                                sign_strike_2nd = template_input['sign_strike_2nd'], 
                                                strike = template_input['min_max_fixed_as_strike'], 
                                                sign_d2_2nd = template_input['sign_d2_2nd'],  
                                                sign_3rd = template_input['sign_3rd'], 
                                                sign_d1_3rd = template_input['sign_d1_3rd'], 
                                                sign_correction_3rd = template_input['sign_correction_3rd'], 
                                                sign_last = template_input['sign_last'],
                                                additional_term = template_input['additional_term'])
                            
        
        return eval(option)
        

    # Option Price
    def _price(self):
        self.opt_type = OptionType.CALL
        call = self._template_pricing_formula()
        
        self.opt_type = OptionType.PUT
        put = self._template_pricing_formula()
        return [call, put]
    
    
    def price(self):
        self._price()
        
 
 