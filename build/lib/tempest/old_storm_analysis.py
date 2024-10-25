import numpy as np
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math 

import warnings
from scipy.optimize import OptimizeWarning
# from sklearn.metrics import UndefinedMetricWarning


# Suppress specific warning

def _piecewise_linear(t:np.array,t_breaks:list,s_max:float):
    """
    Define piecewise linear surface growth over timer with constant value at beginning and end.

    Args:
        t (np.array): t coordinate
        t_breaks (list): t values of break points
        s_breaks (list): surface values of break points

    Returns:
        np.array: piecewize surface
        
    """
    
    N_breaks = len(t_breaks)
    s_breaks = [0, s_max, 0]
    cond_list = [t <= t_breaks[0]]+\
                [np.logical_and(t > t_breaks[i-1],t <= t_breaks[i]) for i in range(1,N_breaks)]+\
                [t > t_breaks[N_breaks-1]]
                
    def make_piece(k):
        def f(t):
            return s_breaks[k-1]+(s_breaks[k]-s_breaks[k-1])/(t_breaks[k]-t_breaks[k-1])*(t-t_breaks[k-1])
        return f 
    
    # Returns 0 rather than s_breaks[0], s_breaks[N_breaks-1] to ensure apparition and disparition of cloud.
    func_list = [lambda t: s_breaks[0]]+\
                [make_piece(k) for k in range(1,N_breaks)]+\
                [lambda t: s_breaks[N_breaks-1]]
                
    return np.piecewise(t,cond_list,func_list)

def _piecewise_fit(t:np.array,s:np.array,t_breaks_0:list,s_max_0:float):    
    """
    Compute piecewise-linear fit of surf(t).

    Args:
        t (np.array): t coordinate
        s (np.array): surface over time
        t_breaks_0 (list): initial t values of break points
        s_max_0(list): maximal surf value

    Returns:
        t_breaks (list): fitted t values of break points
        s_max (list): fitted s_max value
        s_id (np.array): piecewize s fit

    """

    N_breaks = len(t_breaks_0)

    def piecewise_fun(t,*p):
        return _piecewise_linear(t,p[0:N_breaks],p[-1])

    # we add bounds so that time breaks stay ordered
    t_lower_bounds = [-np.inf] + t_breaks_0[:-1]
    t_upper_bounds = t_breaks_0[1:] + [np.inf]
    
    s_lower_bounds = [0] # Positive or null surfaces. 
    s_upper_bounds = [+np.inf] #null surfaces at first and last breaks
    
    p , e = curve_fit(piecewise_fun, t, s, p0=t_breaks_0+[s_max_0], bounds = (t_lower_bounds + s_lower_bounds, t_upper_bounds + s_upper_bounds))

    s_id = _piecewise_linear(t, p[0:N_breaks], p[-1])
    s_max = p[-1]
    t_breaks = list(p[0:N_breaks])
    
    return t_breaks,s_max,s_id

def set_storm_growth_rate(storm, r_treshold = 0.85, verbose = False, plot = False):
    """
    Given a storm object, update it's growth_rate attribute 
    Returns an ax object to plot the fit
    """
    surf = np.array(storm.clusters.surfPix_172Wm2) * 16
    
    if len(surf) <= 4 : 
        growth_rate = np.nan
        setattr(storm, 'growth_rate', growth_rate)
        if verbose : print("A very short-lived storm passed by here...")
        return None
    else : 
        time = np.arange(0, len(surf))

        s_max = max(surf)
        time_breaks = [0, len(surf)//2, len(surf)]

        # warnings.filterwarnings("error", category=UndefinedMetricWarning)
        warnings.filterwarnings("error", category=OptimizeWarning)

        try:
            # Your existing code that raises the warning
            t_breaks, s_max, s_id = _piecewise_fit(time, surf, time_breaks, s_max)
            r_squared = r2_score(surf, s_id)
            growth_r_squared = r2_score(surf[:math.ceil(t_breaks[1])], s_id[:math.ceil(t_breaks[1])])
            decay_r_squared = r2_score(surf[math.floor(t_breaks[1]):], s_id[math.floor(t_breaks[1]):])
            growth_rate = s_max / (t_breaks[1] - t_breaks[0])
            setattr(storm, 'growth_rate', growth_rate)
            
        except OptimizeWarning as e:
            print("That's a complicated storm")
        
        except Exception as e:
            # Handle the warning here, e.g., print a message or log it
            print("Caught Exception:", e)
        
        if verbose : print(f"For storm with label {storm.label}, the growth rate computed by fitting a triangle is {growth_rate} with an r-score of {r_squared}")
    
        return r_squared, growth_r_squared, decay_r_squared, *t_breaks, s_max


