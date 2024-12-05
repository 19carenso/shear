import numpy as np
import matplotlib.pyplot as plt

import xarray as xr
import random 
import os 

from work import handler
from work import casestudy
from work import storm_tracker

settings_path = 'settings/sam3d.yaml'

hdlr = handler.Handler(settings_path)
cs = casestudy.CaseStudy(hdlr, overwrite = False ,verbose = False)
st = storm_tracker.StormTracker(cs, overwrite_storms = False, overwrite = False, verbose = True) #overwrite = True is super long, computes growth rate (triangle fit)


print("bite")
### LOAD STORMS ###


duration_min = 6 #10
surfmaxkm2_min = 10000 #50000 #20000 has 10k elements while 10000 has 29k 
region_latmin, region_latmax, region_lonmin, region_lonmax = -15, 30, -180, 180
# filename_save = f"updated_storms_dmin{duration_min}_smin{surfmaxkm2_min}_lat{region_latmin}_{region_latmax}_lon{region_lonmin}_{region_lonmax}.nc"
# storms_path = os.path.join(os.path.join(st.settings["DIR_DATA_OUT"], cs.name), filename_save)
# ft = xr.open_dataset(storms_path)
storms = xr.open_dataset(st.file_storms)

def filter_storm(ft, region_lonmin, region_lonmax, region_latmin, region_latmax, duration_min, surfmaxkm2_min, vavg_min=None, vavg_max=None): #, start_date =

        # Filter based on duration and surface area
    ft = ft.where(ft.INT_duration > duration_min, drop=True)  # 1min
    ft = ft.where(ft.INT_surfmaxkm2_241K > surfmaxkm2_min, drop=True)
     
    if vavg_max is None and vavg_min is None : 
        pass
    elif vavg_max is None :
        ft =  ft.where(ft.INT_velocityAvg > vavg_min, drop=True)
    elif vavg_min is None :
        ft = ft.where(ft.INT_velocityAvg < vavg_max, drop=True)
    elif vavg_min is not None and vavg_max is not None : 
        print("you don't understand what you're doing my dear")
    # Apply latitude and longitude constraints
    # ft = ft.where((ft.INT_latmin >= region_latmin) & (ft.INT_latmax <= region_latmax), drop=True)
    # if region_lonmin>=0 : 
    #     ft = ft.where((ft.INT_lonmin >= region_lonmin) & (ft.INT_lonmax <= region_lonmax), drop=True)
    # elif region_lonmin<0:
    #     ft = ft.where((ft.INT_lonmin >= 360+region_lonmin) | (ft.INT_lonmin <= region_lonmax), drop=True)
    return ft

ft = filter_storm(storms, region_lonmin, region_lonmax, region_lonmax, region_latmax, duration_min, surfmaxkm2_min)

# ft["INT_max_accumulated_90"] = np.max(ft.LC_accumulated_prec_90th, axis=1)
# ft["INT_max_accumulated_95"] = np.max(ft.LC_accumulated_prec_95th, axis=1)
# ft["INT_sum_total"] = np.sum(ft.LC_total_prec, axis=1)

# ft = ft.dropna(dim='DCS_number', subset=['INT_velocityAvg', 'INT_sum_total'])


################
### -- Filter to only consider DCS that have the 3d fields available at both init and s_max time
################

is_in_X = np.full(len(ft.DCS_number), False)
DCS_numbers = ft.DCS_number.values

for iDCS, DCS_number in enumerate(DCS_numbers):
    start, end, lons, lats, speeds, times, time_smax, i_smax, lons_3d, lats_3d, speeds_3d, times_3d, speed_lon_3d, speed_lat_3d  = st.get_frame_data(ft, DCS_number)
    # DCS = ft.sel(DCS_number=DCS_number)
    if i_smax == -1:
        continue 
    if (times[0] in times_3d):
        print(100 * iDCS / len(DCS_numbers))
        is_in_X[iDCS] = True

ft['is_in_X'] = ('DCS_number', is_in_X)
ft_X = ft.where(ft.is_in_X, drop=True)

print(ft)
### BUILDING HAPPENS HERE ###

# duration_min = 6  # or 10
# surfmaxkm2_min = 20000  # or other value
# region_latmin, region_latmax, region_lonmin, region_lonmax = -15, 30, -180, 180

# Filename and path for saving the dataset
filename_save = f"wind_init_profile_dataset_storms_dmin{duration_min}_smin{surfmaxkm2_min}_lat{region_latmin}_{region_latmax}_lon{region_lonmin}_{region_lonmax}.nc"
storms_path = os.path.join(st.settings["DIR_DATA_OUT"], cs.name, filename_save)

# Initialize lists to collect data
# TABS_init_profiles = []
# QV_init_profiles = []
U_init_profiles = []
V_init_profiles = []

lon_init_list = []
lat_init_list = []
time_init_list = []

# Determine the vertical levels (z)
# Assuming that the vertical levels are the same for all profiles
# We'll get the vertical levels from the first profile
first_DCS_number = ft_X.DCS_number.values[0]
start, end, lons, lats, speeds, times, time_smax, i_smax, lons_3d, lats_3d, speeds_3d, times_3d, speed_lon_3d, speed_lat_3d = st.get_frame_data(ft_X, first_DCS_number)
DCS = ft.sel(DCS_number=first_DCS_number)

lon_init = lons[0]
lat_init = lats[0]
time_init = times[0]
extent_init, slice_lon_init, slice_lat_init = st.get_full_extent_slice([lon_init], [lat_init], large_scale_frame_size=0.7)

# Get vertical levels from the initial profile
le_dico_init = {"lat": slice_lat_init, "lon": slice_lon_init}
# # # TABS_init_profile = hdlr.load_var(cs, "TABS", i_t=time_init, z_idx="L'altitude de la troposphère", sel_dict=le_dico_init).TABS[0].mean(dim=['lat', 'lon']).values
z_levels = hdlr.load_var(cs, "TABS", i_t=time_init, z_idx="L'altitude de la troposphère", sel_dict=le_dico_init).z.values

num_DCS = len(ft_X.DCS_number)
num_z = len(z_levels)

n_DCS = len(ft_X.DCS_number.values)

# Initialize arrays to store profiles
# TABS_init_array = np.full((num_DCS, num_z), np.nan)
# QV_init_array = np.full((num_DCS, num_z), np.nan)
U_init_array = np.full((num_DCS, num_z), np.nan)
V_init_array = np.full((num_DCS, num_z), np.nan)

# Loop over each DCS_number to collect data

for iDCS, DCS_number in enumerate(ft_X.DCS_number.values): #[:third]
    print(100*iDCS/len(ft_X.DCS_number.values))
    # Retrieve data
    start, end, lons, lats, speeds, times, time_smax, i_smax, lons_3d, lats_3d, speeds_3d, times_3d, speed_lon_3d, speed_lat_3d = st.get_frame_data(ft_X, DCS_number)
    DCS = ft.sel(DCS_number=DCS_number)

    # Initial positions and times
    lon_init = lons[0]
    lat_init = lats[0]
    time_init = times[0]
    extent_init, slice_lon_init, slice_lat_init = st.get_full_extent_slice([lon_init], [lat_init], large_scale_frame_size=0.7)
    
    # Dictionaries for data selection
    le_dico_init = {"lat": slice_lat_init, "lon": slice_lon_init}
    
    # Load initial profiles
    # # # TABS_init_profile = hdlr.load_var(cs, "TABS", i_t=time_init, z_idx="L'altitude de la troposphère", sel_dict=le_dico_init).TABS[0].mean(dim=['lat', 'lon']).values
    # # # QV_init_profile = hdlr.load_var(cs, "QV", i_t=time_init, z_idx="L'altitude de la troposphère", sel_dict=le_dico_init).QV[0].mean(dim=['lat', 'lon']).values
    U_init_profile = hdlr.load_var(cs, "U", i_t=time_init, z_idx="L'altitude de la troposphère", sel_dict=le_dico_init).U[0].mean(dim=['lat', 'lon']).values
    V_init_profile = hdlr.load_var(cs, "V", i_t=time_init, z_idx="L'altitude de la troposphère", sel_dict=le_dico_init).V[0].mean(dim=['lat', 'lon']).values
    
    # Store profiles in arrays
    # # TABS_init_array[iDCS, :] = TABS_init_profile
    # # QV_init_array[iDCS, :] = QV_init_profile
    U_init_array[iDCS, :] = U_init_profile
    V_init_array[iDCS, :] = V_init_profile
    
    # Store positions and times
    lon_init_list.append(lon_init)
    lat_init_list.append(lat_init)
    time_init_list.append(time_init)
    
# Convert lists to numpy arrays
lon_init_array = np.array(lon_init_list)
lat_init_array = np.array(lat_init_list)
time_init_array = np.array(time_init_list)

# Create the xarray Dataset
ds = xr.Dataset(
    {
        # Initial profiles
        # # 'TABS_init_profile': (('DCS_number', 'z'), TABS_init_array),
        # # 'QV_init_profile': (('DCS_number', 'z'), QV_init_array),
        'U_init_profile': (('DCS_number', 'z'), U_init_array),
        'V_init_profile': (('DCS_number', 'z'), V_init_array),

        'lon_init': (('DCS_number',), lon_init_array),
        'lat_init': (('DCS_number',), lat_init_array),
        'time_init': (('DCS_number',), time_init_array),
    },
    coords={
        'DCS_number': ft_X.DCS_number.values,
        'z': z_levels,
    }
)

ds.to_netcdf(storms_path)
