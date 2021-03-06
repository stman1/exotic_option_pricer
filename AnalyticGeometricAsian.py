# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 12:48:16 2022

@author: Stefan Mangold
"""

from numpy import *
from scipy.stats import norm

class ClosedFormGeometricAsian:
    
    """
    This is a class for geometric Asian options contracts on stocks without dividends.
    
    Attributes: 
        spot            : int or float
        strike          : int or float 
        rate            : float
        dte             : int or float [days to expiration in number of years]
        volatility      : float
        num_avg_samples : int
    """    
    
    def __init__(self, spot, strike, rate, dte, volatility, num_avg_samples):
        
        # Spot Price
        self.spot = spot
        
        # Option Strike
        self.strike = strike
        
        # Interest Rate
        self.rate = rate
        
        # Days To Expiration
        self.dte = dte
        
        # Volatility
        self.volatility = volatility
        
        # Number of sample observations taken in building average
        self.avg_sample = num_avg_samples
        
        # geometric asian volatility
        self.vol_asian = self.volatility *\
            sqrt(((self.avg_sample + 1)*(2 * self.avg_sample + 1))/(6 * (self.avg_sample)**2))
        
        # geometric asian drift under martingale measure
        self.drift_asian = 0.5 * self.vol_asian + \
            (self.rate - 0.5 * (self.volatility)**2)*(self.avg_sample + 1)/(2 * self.avg_sample + 1)

        # Utility 
        self._a_ = self.vol_asian * self.dte**0.5
        
        if self.strike == 0:
            raise ZeroDivisionError('The strike price cannot be zero') # raise an exception
        else:
            self._d1_ = (log(self.spot / self.strike) + \
                     (self.drift_asian + (self.vol_asian**2) / 2) * self.dte) / self._a_
        
        self._d2_ = self._d1_ - self._a_
        
        self._b_ = e**-(self.rate * self.dte)
        
    
        [self.call_price, self.put_price] = self._price()

        
    # Option Price
    def _price(self):
        '''Returns the option price: [Call price, Put price]'''

        if self.volatility == 0 or self.dte == 0:
            call = maximum(0.0, self.spot - self.strike)
            put = maximum(0.0, self.strike - self.spot)
        else:
            call = self.spot * norm.cdf(self._d1_) - self.strike * e**(-self.rate * \
                                                                       self.dte) * norm.cdf(self._d2_)

            put = self.strike * e**(-self.rate * self.dte) * norm.cdf(-self._d2_) - \
                                                                        self.spot * norm.cdf(-self._d1_)
        return [call, put]
