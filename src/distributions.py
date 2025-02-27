import numpy as np
from scipy.stats import invgamma, gamma, norm, uniform
from scipy.optimize import bisect
import scipy.special as f

"""

    IMPLEMENT THE SKEWED GENERALIZED T DISTRIBUTION

    Here we assume that beta = 2 and also that mu=0, sigma=1
    and follow https://arxiv.org/pdf/2401.14122.

    The parameter a > 0 controls the tail,
    it has all p moments with p in [0,2a).

    The parameter r must satisfy |r| < 1.

    Notice that:
    - if a = np.inf and r = 0 it is the Gaussian;
    - if a = v/2 and it is a student with v degrees of freedom;
    - if a = .5 and r = 0 it is a Cauchy.    

"""

class SkeGTD:
    def __init__(self, a, r, rng):
        self.a = a
        self.r = r
        
        invgamma.random_state = rng
        gamma.random_state = rng
        norm.random_state = rng
        uniform.random_state = rng
    

    def rvs(self, shape):
        if self.a == np.inf:
            if self.r == 0:
                # is a Gaussian
                return norm.rvs(size = shape)
            else:
                # is a skewed Gaussian
                W = uniform.rvs(size = shape) < (self.r+1)/2
                W = (self.r-1) + 2*W
                Y = gamma.rvs(.5, size=shape)               
                return np.sqrt(2)*np.sqrt(Y)*W
        else:
            W = uniform.rvs(size = shape) < (self.r+1)/2
            W = (self.r-1) + 2*W
            Y = gamma.rvs(.5, size=shape)
            Z = invgamma.rvs(self.a, scale=self.a, size = shape)              
            return np.sqrt(2)*np.sqrt(Y)*W*np.sqrt(Z)

    
    def mean(self):
        if self.a == np.inf:
            if self.r == 0:
                # is a Gaussian
                return 0
            else:
                # is a skewed Gaussian             
                return np.sqrt(2/np.pi)*2*self.r
        else:
            if self.a <= .5:
                return np.inf
            else:
                return np.sqrt(2*self.a/np.pi)*2*self.r * f.gamma(self.a - .5)/f.gamma(self.a)
    

    def var(self):
        if self.a == np.inf:
            if self.r == 0:
                # is a Gaussian
                return 1
            else:
                # is a skewed Gaussian             
                return (1+3*self.r**2) - self.mean()**2
        else:
            if self.a <= 1:
                return np.inf
            else:
                return (1+3*self.r**2)*self.a/(self.a-1) - self.mean()**2
    

    def std(self):
        return np.sqrt(self.var())
    

    def cdf(self, x):
        if self.a == np.inf:
            if self.r == 0:
                # is a Gaussian
                return norm.cdf(x)
            else:
                # is a skewed Gaussian
                xneg = (1-self.r)/2 * (1 - gamma.cdf(x**2/(2*(1-self.r)**2), .5))
                xpos = (1-self.r)/2 + (self.r+1)/2*gamma.cdf( x**2/(2*(self.r+1)**2), .5)
                return (x < 0)*xneg + (x >=0)*xpos
        else:
            u = 1/ (1 + x**2 / ( 2*self.a*( 1 + self.r * np.sign(x) )**2 ))
            xpos = 1 - (self.r + 1)/2 * f.betainc( self.a, .5, u )
            xneg = ( 1 - self.r)/2 * f.betainc( self.a, .5, u )
            return (x < 0)*xneg + (x >=0)*xpos

            
    def median(self):
        # find upper and lower values
        a = -1
        b = 1
        while self.cdf(a) > 1/2:
            a *= 2
        while self.cdf(b) < 1/2:
            b *= 2
        return bisect(lambda x: self.cdf(x)-.5, a, b)
