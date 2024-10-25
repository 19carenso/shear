import numpy as np

#########################################################################
##  Dry air                                                            ##
c_vd             = 719.             # J/kg/K                           ##
R_d              = 287.04           # J/kg/K                           ##
c_pd             = c_vd + R_d       # J/kg/K                           ##
                                                                       ##
##  Water vapor                                                        ##
c_vv             = 1418.            # J/kg/K                           ##
R_v              = 461.4            # J/kg/K                           ##
c_pv             = c_vv + R_v       # J/kg/K                           ##
                                                                       ##
##  Liquid water                                                       ##
c_vl             = 4216.            # J/kg/K                           ##
                                                                       ##
##  Solid water                                                        ##
c_vs             = 2106.            # J/kg/K                           ##    
                                                                       ##
##  Reference temperatures and pressures                               ##
T_0              = 273.16           # K                                ## 
p_0              = 1.e5             # Pa                               ##
e_0              = 611.65           # Pa                               ##
                                                                       ##
##  Energies, enthalpies, entropies                                    ##
L_v              = 2260000.         # J/kg                             ##
L_cond           = 2.5104e6         # J/kg consistently with SAM       ##
E_0v             = 2374000.         # J/kg                             ##
E_0s             = 333700.          # J/kg                             ##
s_0v             = E_0v/T_0 + R_v   # J/kg/K                           ##
s_0s             = E_0s/T_0         # J/kg/K                           ##
                                                                       ##
##  other                                                              ##
gg               = 9.81             # m/s^2                            ##
eps              = R_d/R_v          # Unitless                         ##
                                                                       ##
## Densities                                                           ##
rho_l			 = 998.23           # density of water, kg/m3          ##
#########################################################################

def saturation_vapor_pressure(temp):

    """Argument: Temperature (K) as a numpy.ndarray or dask.array
    Returns: saturation vapor pressure (Pa) in the same format."""

    T_0 = 273.15
    cn = np
    
    def qvstar_numpy(temp):

        whereAreNans = np.isnan(temp)
        temp_wo_Nans = temp.copy()
        temp_wo_Nans[whereAreNans] = 0.
        # Initialize
        e_sat = np.zeros(temp.shape)
        e_sat[whereAreNans] = np.nan
        #!!! T > 0C
        overliquid = (temp_wo_Nans > T_0)
        ## Buck
        e_sat_overliquid = 611.21*np.exp(np.multiply(18.678-(temp-T_0)/234.5,
                                                      np.divide((temp-T_0),257.14+(temp-T_0))))
        # ## Goff Gratch equation  for liquid water below 0ºC
        # e_sat_overliquid = np.power(10,-7.90298*(373.16/temp-1)
        #             + 5.02808*np.log10(373.16/temp) 
        #             - 1.3816e-7*(np.power(10,11.344*(1-temp/373.16)) -1) 
        #            + 8.1328e-3*(np.power(10,-3.49149*(373.16/temp-1)) -1) 
        #            + np.log10(1013.246)) 
        e_sat[overliquid] = e_sat_overliquid[overliquid]
        #!!! T < 0C 
        overice = (temp_wo_Nans < T_0)
        # ## Buck
        # e_sat_overice = 611.15*np.exp(np.multiply(23.036-(temp-T_0)/333.7,
        #                                            np.divide((temp-T_0),279.82+(temp-T_0))))
        ## Goff Gratch equation over ice 
        e_sat_overice =  100*np.power(10,-9.09718*(273.16/temp - 1) 
                    - 3.56654*np.log10(273.16/temp) 
                    + 0.876793*(1 - temp/ 273.16) 
                    + np.log10(6.1071))
        e_sat[overice] = e_sat_overice[overice]

        return e_sat       # in Pa

    if cn is np:
        return qvstar_numpy(temp)
#    elif temp.__class__ == da.core.Array:
#        return da.map_blocks(qvstar_numpy,temp,dtype=np.float64)
    elif 'float' in str(temp.__class__):
        if temp > T_0:
            return 611.21*np.exp((18.678-(temp-T_0)/234.5)*(temp-T_0)/(257.14+(temp-T_0)))
        else:
            return 611.15*np.exp((23.036-(temp-T_0)/333.7)*(temp-T_0)/(279.82+(temp-T_0)))
    else:
        print("[Error in thermoFunctions.saturationVaporPressure] Unvalid data type:", type(temp))
        return
    
def saturation_specific_humidity(temp,pres):

    """Convert from estimate of saturation vapor pressure to saturation specific
    humidity using the approximate equation qvsat ~ epsilon"""

    e_sat = saturation_vapor_pressure(temp)
    qvstar = (e_sat/R_v)/(pres/R_d)

    return qvstar

def haversine(lon1, lat1, lon2, lat2, R=6371000):
    """
    Calculate the great-circle distance between two points 
    on the Earth using the Haversine formula.
    
    Parameters:
    - lon1, lat1: Longitude and latitude of the first point in degrees.
    - lon2, lat2: Longitude and latitude of the second point in degrees.
    - R: Radius of the Earth in meters.
    
    Returns:
    - distance: The distance between the two points in meters.
    """
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c