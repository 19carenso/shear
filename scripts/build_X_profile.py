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


### LOAD STORMS ###

duration_min = 6 #10
surfmaxkm2_min = 10000 #50000 #20000 has 10k elements while 10000 has 29k 
region_latmin, region_latmax, region_lonmin, region_lonmax = -15, 30, -180, 180
filename_save = f"updated_storms_dmin{duration_min}_smin{surfmaxkm2_min}_lat{region_latmin}_{region_latmax}_lon{region_lonmin}_{region_lonmax}.nc"
storms_path = os.path.join(os.path.join(st.settings["DIR_DATA_OUT"], cs.name), filename_save)
ft = xr.open_dataset(storms_path)
storms = xr.open_dataset(st.file_storms)

def filter_storm(ft, region_lonmin, region_lonmax, region_latmin, region_latmax, vavg_min=None, vavg_max=None): #, start_date = 
    if vavg_max is None and vavg_min is None : 
        pass
    elif vavg_max is None :
        ft =  ft.where(ft.INT_velocityAvg > vavg_min, drop=True)
    elif vavg_min is None :
        ft = ft.where(ft.INT_velocityAvg < vavg_max, drop=True)
    elif vavg_min is not None and vavg_max is not None : 
        print("you don't understand what you're doing my dear")
    # Apply latitude and longitude constraints
    ft = ft.where((ft.INT_latmin >= region_latmin) & (ft.INT_latmax <= region_latmax), drop=True)
    if region_lonmin>=0 : 
        ft = ft.where((ft.INT_lonmin >= region_lonmin) & (ft.INT_lonmax <= region_lonmax), drop=True)
    elif region_lonmin<0:
        ft = ft.where((ft.INT_lonmin >= 360+region_lonmin) | (ft.INT_lonmin <= region_lonmax), drop=True)
    return ft


ft["INT_max_accumulated_90"] = np.max(ft.LC_accumulated_prec_90th, axis=1)
ft["INT_max_instant_99"] = np.max(ft.LC_instant_prec_99th, axis=1)
ft["INT_max_accumulated_95"] = np.max(ft.LC_accumulated_prec_95th, axis=1)
ft["INT_max_instant_95"] = np.max(ft.LC_instant_prec_95th, axis=1)
ft["INT_sum_total"] = np.sum(ft.LC_total_prec, axis=1)

ft = ft.dropna(dim='DCS_number', subset=['INT_velocityAvg', 'INT_sum_total', 'INT_max_instant_99', 'INT_max_accumulated_90'])

# xt =  ft.where((ft.INT_max_accumulated_95)+2*(ft.INT_max_instant_99)>200, drop=True)

###

is_in_X = np.full(len(ft.DCS_number), False)
DCS_numbers = ft.DCS_number.values
for iDCS, DCS_number in enumerate(DCS_numbers):
    start, end, lons, lats, speeds, times, time_smax, i_smax, lons_3d, lats_3d, speeds_3d, times_3d, speed_lon_3d, speed_lat_3d  = st.get_frame_data(ft, DCS_number)
    DCS = ft.sel(DCS_number=DCS_number)
    if i_smax == -1:
        continue 
    LC_instant_prec_99th = DCS.LC_instant_prec_99th.values[start:end][:i_smax+1]
    INT_max_instant_99 = DCS.INT_max_instant_99.values
    mask_max_instant_99 = LC_instant_prec_99th == INT_max_instant_99
    if not np.any(mask_max_instant_99):
        continue  
    time_max_instant_99 = times[:i_smax+1][mask_max_instant_99][0]
    if (times[0] in times_3d) and (time_max_instant_99 in times_3d) and (times[0] != time_max_instant_99): # i want the two times to be different
        print(100 * iDCS / len(DCS_numbers))
        is_in_X[iDCS] = True

ft['is_in_X'] = ('DCS_number', is_in_X)
ft_X = ft.where(ft.is_in_X, drop=True)


### BUILDING HAPPENS HERE ###

duration_min = 6  # or 10
surfmaxkm2_min = 10000  # or other value
region_latmin, region_latmax, region_lonmin, region_lonmax = -15, 30, -180, 180

# Filename and path for saving the dataset
filename_save = f"profile_dataset_storms_dmin{duration_min}_smin{surfmaxkm2_min}_lat{region_latmin}_{region_latmax}_lon{region_lonmin}_{region_lonmax}.nc"
storms_path = os.path.join(st.settings["DIR_DATA_OUT"], cs.name, filename_save)

# Initialize lists to collect data
TABS_init_profiles = []
QV_init_profiles = []
U_init_profiles = []
V_init_profiles = []

TABS_max_instant_profiles = []
QV_max_instant_profiles = []
U_max_instant_profiles = []
V_max_instant_profiles = []

lon_init_list = []
lat_init_list = []
time_init_list = []

lon_max_instant_list = []
lat_max_instant_list = []
time_max_instant_list = []

# Determine the vertical levels (z)
# Assuming that the vertical levels are the same for all profiles
# We'll get the vertical levels from the first profile
first_DCS_number = ft_X.DCS_number.values[0]
start, end, lons, lats, speeds, times, time_smax, i_smax, lons_3d, lats_3d, speeds_3d, times_3d, speed_lon_3d, speed_lat_3d = st.get_frame_data(ft_X, first_DCS_number)
DCS = ft.sel(DCS_number=first_DCS_number)
mask_max_instant_99 = (DCS.LC_instant_prec_99th[start:end][:i_smax+1] == DCS.INT_max_instant_99)
time_max_instant_99 = times[:i_smax+1][mask_max_instant_99][0]

lon_init = lons[0]
lat_init = lats[0]
time_init = times[0]
extent_init, slice_lon_init, slice_lat_init = st.get_full_extent_slice([lon_init], [lat_init], large_scale_frame_size=0.7)

# Get vertical levels from the initial profile
le_dico_init = {"lat": slice_lat_init, "lon": slice_lon_init}
TABS_init_profile = hdlr.load_var(cs, "TABS", i_t=time_init, z_idx="L'altitude de la troposphère", sel_dict=le_dico_init).TABS[0].mean(dim=['lat', 'lon']).values
z_levels = hdlr.load_var(cs, "TABS", i_t=time_init, z_idx="L'altitude de la troposphère", sel_dict=le_dico_init).z.values

num_DCS = len(ft_X.DCS_number)
num_z = len(z_levels)

# Initialize arrays to store profiles
TABS_init_array = np.full((num_DCS, num_z), np.nan)
QV_init_array = np.full((num_DCS, num_z), np.nan)
U_init_array = np.full((num_DCS, num_z), np.nan)
V_init_array = np.full((num_DCS, num_z), np.nan)

TABS_max_instant_array = np.full((num_DCS, num_z), np.nan)
QV_max_instant_array = np.full((num_DCS, num_z), np.nan)
U_max_instant_array = np.full((num_DCS, num_z), np.nan)
V_max_instant_array = np.full((num_DCS, num_z), np.nan)

# Loop over each DCS_number to collect data
for iDCS, DCS_number in enumerate(ft_X.DCS_number.values):
    print(100*iDCS/len(ft_X.DCS_number.values))
    # Retrieve data
    start, end, lons, lats, speeds, times, time_smax, i_smax, lons_3d, lats_3d, speeds_3d, times_3d, speed_lon_3d, speed_lat_3d = st.get_frame_data(ft_X, DCS_number)
    DCS = ft.sel(DCS_number=DCS_number)
    mask_max_instant_99 = (DCS.LC_instant_prec_99th[start:end][:i_smax+1] == DCS.INT_max_instant_99)
    time_max_instant_99 = times[:i_smax+1][mask_max_instant_99][0]
    
    # Initial positions and times
    lon_init = lons[0]
    lat_init = lats[0]
    time_init = times[0]
    extent_init, slice_lon_init, slice_lat_init = st.get_full_extent_slice([lon_init], [lat_init], large_scale_frame_size=0.7)
    
    # Positions and times at maximum instant
    lon_max_instant = lons[:i_smax+1][mask_max_instant_99][0]
    lat_max_instant = lats[:i_smax+1][mask_max_instant_99][0]
    time_max_instant = times[:i_smax+1][mask_max_instant_99][0]
    extent_max_instant, slice_lon_max_instant, slice_lat_max_instant = st.get_full_extent_slice([lon_max_instant], [lat_max_instant], large_scale_frame_size=0.7)
    
    # Dictionaries for data selection
    le_dico_init = {"lat": slice_lat_init, "lon": slice_lon_init}
    le_dico_max_instant = {"lat": slice_lat_max_instant, "lon": slice_lon_max_instant}
    
    # Load initial profiles
    TABS_init_profile = hdlr.load_var(cs, "TABS", i_t=time_init, z_idx="L'altitude de la troposphère", sel_dict=le_dico_init).TABS[0].mean(dim=['lat', 'lon']).values
    QV_init_profile = hdlr.load_var(cs, "QV", i_t=time_init, z_idx="L'altitude de la troposphère", sel_dict=le_dico_init).QV[0].mean(dim=['lat', 'lon']).values
    U_init_profile = hdlr.load_var(cs, "U", i_t=time_init, z_idx="L'altitude de la troposphère", sel_dict=le_dico_init).U[0].mean(dim=['lat', 'lon']).values
    V_init_profile = hdlr.load_var(cs, "V", i_t=time_init, z_idx="L'altitude de la troposphère", sel_dict=le_dico_init).V[0].mean(dim=['lat', 'lon']).values
    
    # Load profiles at maximum instant
    TABS_max_instant_profile = hdlr.load_var(cs, "TABS", i_t=time_max_instant, z_idx="L'altitude de la troposphère", sel_dict=le_dico_max_instant).TABS[0].mean(dim=['lat', 'lon']).values
    QV_max_instant_profile = hdlr.load_var(cs, "QV", i_t=time_max_instant, z_idx="L'altitude de la troposphère", sel_dict=le_dico_max_instant).QV[0].mean(dim=['lat', 'lon']).values
    U_max_instant_profile = hdlr.load_var(cs, "U", i_t=time_max_instant, z_idx="L'altitude de la troposphère", sel_dict=le_dico_max_instant).U[0].mean(dim=['lat', 'lon']).values
    V_max_instant_profile = hdlr.load_var(cs, "V", i_t=time_max_instant, z_idx="L'altitude de la troposphère", sel_dict=le_dico_max_instant).V[0].mean(dim=['lat', 'lon']).values
    
    # Store profiles in arrays
    TABS_init_array[iDCS, :] = TABS_init_profile
    QV_init_array[iDCS, :] = QV_init_profile
    U_init_array[iDCS, :] = U_init_profile
    V_init_array[iDCS, :] = V_init_profile
    
    TABS_max_instant_array[iDCS, :] = TABS_max_instant_profile
    QV_max_instant_array[iDCS, :] = QV_max_instant_profile
    U_max_instant_array[iDCS, :] = U_max_instant_profile
    V_max_instant_array[iDCS, :] = V_max_instant_profile
    
    # Store positions and times
    lon_init_list.append(lon_init)
    lat_init_list.append(lat_init)
    time_init_list.append(time_init)
    
    lon_max_instant_list.append(lon_max_instant)
    lat_max_instant_list.append(lat_max_instant)
    time_max_instant_list.append(time_max_instant)

# Convert lists to numpy arrays
lon_init_array = np.array(lon_init_list)
lat_init_array = np.array(lat_init_list)
time_init_array = np.array(time_init_list, dtype='datetime64[ns]')

lon_max_instant_array = np.array(lon_max_instant_list)
lat_max_instant_array = np.array(lat_max_instant_list)
time_max_instant_array = np.array(time_max_instant_list, dtype='datetime64[ns]')

# Create the xarray Dataset
ds = xr.Dataset(
    {
        # Initial profiles
        'TABS_init_profile': (('DCS_number', 'z'), TABS_init_array),
        'QV_init_profile': (('DCS_number', 'z'), QV_init_array),
        'U_init_profile': (('DCS_number', 'z'), U_init_array),
        'V_init_profile': (('DCS_number', 'z'), V_init_array),
        # Profiles at maximum instant
        'TABS_max_instant_profile': (('DCS_number', 'z'), TABS_max_instant_array),
        'QV_max_instant_profile': (('DCS_number', 'z'), QV_max_instant_array),
        'U_max_instant_profile': (('DCS_number', 'z'), U_max_instant_array),
        'V_max_instant_profile': (('DCS_number', 'z'), V_max_instant_array),
        # Positions and times
        'lon_init': (('DCS_number',), lon_init_array),
        'lat_init': (('DCS_number',), lat_init_array),
        'time_init': (('DCS_number',), time_init_array),
        'lon_max_instant': (('DCS_number',), lon_max_instant_array),
        'lat_max_instant': (('DCS_number',), lat_max_instant_array),
        'time_max_instant': (('DCS_number',), time_max_instant_array),
    },
    coords={
        'DCS_number': ft_X.DCS_number.values,
        'z': z_levels,
    }
)

# Save the dataset
ds.to_netcdf(storms_path)
