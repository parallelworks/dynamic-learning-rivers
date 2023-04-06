#================================================
# Define function to compute saturated oxygen
#================================================
def sw_o2sat(sa,te):
    
    import numpy as np
    
    # This function computes saturated O2 based on salinity (sa)
    # and temperature (te).  You can use in situ temperature rather
    # than potential temperature.
    #
    # ***---> The output is in mL O2/L seawater! <---***
    #
    # The original code was written by Phil Morgan and Lindsay Pender
    # (Lindsay.Pender@csiro.au) as part of the CISRO Seawater Toolbox
    # based on the equations determined by:
    # Weiss, R. F. 1970
    # "The solubility of nitrogen, oxygen and argon in water and seawater."
    # Deap-Sea Research., 1970, Vol 17, pp721-735.
    # Stefan Gary translated the CISRO sw_satO2 code from MATLAB to Python.
    
    # convert T to Kelvin
    tek = 273.15 + te

    # constants for Eqn (4) of Weiss 1970
    a1 = -173.4292
    a2 =  249.6339
    a3 =  143.3483
    a4 =  -21.8492
    b1 =   -0.033096
    b2 =    0.014259
    b3 =   -0.0017000

    # Eqn (4) of Weiss 1970
    # Numpy operations are not required for additions/subtractions
    # because arrays are implicitly same size.  Numpy operations
    # are not required for scalar multiplicative factors.
    lnC = a1 + a2*np.divide(100,tek) + a3*np.log(tek/100) + a4*(tek/100) + \
        np.multiply(sa,(b1 + b2*(tek/100) + b3*(np.power((tek/100),2) )))

    return np.exp(lnC)
