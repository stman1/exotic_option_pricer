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
        running_max   : int or float
        running_min   : int or float
    """    
    
    def __init__(self, spot, option_type, strike_type, rate, dte, volatility, strike, running_max, running_min):
        
        # Spot Price
        self.spot = spot
        
        # largest asset value observed since contract outset
        self.running_max = running_max
        
        # smallest asset value observed since contract outset
        self.running_min = running_min
        
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
        
        # Volaitlity
        self.volatility = volatility
       
        # Utility 
        self._a_ = self.volatility * self.dte**0.5
        
        if self.strike == 0:
            raise ZeroDivisionError('The strike price cannot be zero') # raise an exception
        else:
            self._d1_ = (log(self.spot / self.strike) + \
                     (self.rate + (self.volatility**2) / 2) * self.dte) / self._a_
        
        self._d2_ = self._d1_ - self._a_
        
        self._b_ = e**-(self.rate * self.dte)
        
        
        # The __dict__ attribute
        '''
        Contains all the attributes defined for the object itself. It maps the attribute name to its value.
        '''
        for i in ['callPrice', 'putPrice']:
            self.__dict__[i] = None
        
        [self.callPrice, self.putPrice] = self._price()


    def _set_d1(self, min_max_strike):
        self._d1_ = (log(self.spot / min_max_strike) + (self.rate + 0.5 * self.volatility**2)*self.dte) / (self.volatility * sqrt(self.dte))

    
    def _template_pricing_formula(self):
                
        call = Template('$sign_spot_1st self.spot * norm.cdf($sign_d1_1st self._d1_)\
                        $sign_strike_2nd $strike * e**(-self.rate * self.dte)\
                        * norm.cdf($sign_d2_2nd self._d2_) +self.spot * e**(-self.rate * self.dte)\
                        * self.volatility**2 / (2*(self.rate)) * ($sign_3rd (self.spot/$strike)**(-(2*self.rate/(self.volatility**2)))\
                        * norm.cdf($sign_d1_3rd self._d1_ $sign_correction_3rd (2*self.rate*sqrt(self.dte))/(self.volatility))\
                        $sign_last e**(self.rate*self.dte) * norm.cdf(-self._d1_)) ')

        s = call.substitute(sign_spot_1st = '+', 
                            sign_d1_1st ='+', 
                            sign_strike_2nd = '-', 
                            strike = 'self.running_min', 
                            sign_d2_2nd = '+',  
                            sign_3rd = '+', 
                            sign_d1_3rd = '-', 
                            sign_correction_3rd = '+', 
                            sign_last = '-')
        
        price = eval(s)
        return price
        

    # Option Price
    def _price(self):
        '''Returns the option price: [Call price, Put price]'''

        if self.volatility == 0 or self.dte == 0:
            call = maximum(0.0, self.spot - self.strike)
            put = maximum(0.0, self.strike - self.spot)
            
        if self.st_type == StrikeType.FLOATING:
            
            if self.opt_type == OptionType.CALL:
                self._set_d1(self.running_min)
                
            elif self.opt_type == OptionType.PUT:
                self._set_d1(self.running_max)
                               
            else:
                raise TypeError(f'Lookback option type is {self.opt_type} but must be either OptionType.CALL or OptionType.PUT') # raise an exception
                            
        elif self.st_type == StrikeType.FIXED:   
            
            if self.opt_type == OptionType.CALL:
                if self.running_max > self.strike:
                    self._set_d1(self.running_max)
                    
                elif self.running_max < self.strike:
                    self._set_d1(self.strike) 
                    
                else:
                    call = 0 
                        
            elif self.opt_type == OptionType.PUT:
                
                if self.running_min > self.strike:
                    self._set_d1(self.strike) 
                 
                elif self.running_min < self.strike:
                    self._set_d1(self.running_max)
                else:
                    put = 0    
                
            else:
                raise TypeError(f'Lookback option type is {self.opt_type} but must be either OptionType.CALL or OptionType.PUT') # raise an exception
            
        
        else:
            raise TypeError(f'Lookback strike type is {self.st_type} but must be either StrikeType.FLOATING or StrikeType.FIXED') # raise an exception
                
        # d2 is always the same
        self._d2_ = self._d1_ - self.volatility * sqrt(self.dte)        
        

        if self.st_type == StrikeType.FLOATING:
            pass    

            # call = self.spot * norm.cdf(self._d1_)\
            #     - self.running_min * e**(-self.rate * self.dte) * norm.cdf(self._d2_)\
            #         +self.spot * e**(-self.rate * self.dte) * self.volatility**2 / (2*(self.rate))\
            #             *((self.spot/self.running_min)**(-(2*self.rate/(self.volatility**2)))\
            #               *norm.cdf(-self._d1_ + (2*self.rate*sqrt(self.dte))/(self.volatility)) - e**(self.rate*self.dte) * norm.cdf(-self._d1_)   )

        elif self.st_type == StrikeType.FIXED:
            pass
        
        else:
            raise TypeError(f'Lookback strike type is {self.st_type} but must be either StrikeType.FLOATING or StrikeType.FIXED') # raise an exception
                    

        
            
        if self.volatility == 0 or self.dte == 0:
            call = maximum(0.0, self.spot - self.strike)
            put = maximum(0.0, self.strike - self.spot)
        else:
            call = self.spot * norm.cdf(self._d1_) - self.strike * e**(-self.rate * \
                                                                       self.dte) * norm.cdf(self._d2_)

            put = self.strike * e**(-self.rate * self.dte) * norm.cdf(-self._d2_) - \
                                                                        self.spot * norm.cdf(-self._d1_)
        return [call, put]

 