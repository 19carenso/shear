import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import random 
import os 


from work import handler
from work import casestudy
from work import storm_tracker

from work.plots.hist import simple_hist
from work.thermo import haversine 

settings_path = 'settings/sam3d.yaml'
# Instantiate CaseStudy by passing the settings. 
# Should also create appropriate directories
hdlr = handler.Handler(settings_path)
cs = casestudy.CaseStudy(hdlr, overwrite = False ,verbose = False)
st = storm_tracker.StormTracker(cs, overwrite_storms = False, overwrite = False, verbose = True) #overwrite = True is super long, computes growth rate (triangle fit)


duration_min = 6
surfmaxkm2_min = 10000
region_latmin, region_latmax, region_lonmin, region_lonmax = -15, 30, -180, 180
save_storms_path = st.settings["DIR_DATA_OUT"] + f"save_storms_dmin{duration_min}_smin{surfmaxkm2_min}_lat{region_latmin}_{region_latmax}_lon{region_lonmin}_{region_lonmax}.nc"


if not os.path.exists(save_storms_path): 
    storms = xr.open_dataset(st.file_storms)
    
    # Filter based on duration and surface area
    storms = storms.where(storms.INT_duration > duration_min, drop=True)  # 1min
    storms = storms.where(storms.INT_surfmaxkm2_241K > surfmaxkm2_min, drop=True)
    
    # Apply latitude and longitude constraints
    storms = storms.where((storms.INT_latmin >= region_latmin) & (storms.INT_latmax <= region_latmax), drop=True)
    if region_lonmin>=0 : 
        storms = storms.where((storms.INT_lonmin >= region_lonmin) & (storms.INT_lonmax <= region_lonmax), drop=True)
    elif region_lonmin<0:
        storms = storms.where((storms.INT_lonmin >= 360+region_lonmin) | (storms.INT_lonmin <= region_lonmax), drop=True)
    
    # Save the filtered storms data
    storms.to_netcdf(save_storms_path)
    storms.close()
else: 
    storms = xr.open_dataset(save_storms_path)


def add_variable_to_dataset(ds, new_data, var_name, dims):
    """    
    Adds a new variable to an xarray.Dataset, supporting both single- and multi-dimensional data.
    
    Parameters:
    ds (xarray.Dataset): The original dataset.
    new_data (numpy.ndarray): The array to be added as a new variable. Its shape should match the provided dims.
    var_name (str): The name of the new variable.
    dims (tuple or str): The dimensions of the new variable (e.g., 'DCS_number' or ('DCS_number', 'time')).
    
    Returns:
    xarray.Dataset: The dataset with the new variable added.
    """
    # Ensure dims is a tuple if only one dimension is provided
    if isinstance(dims, str):
        dims = (dims,)
    
    # Create a dictionary of coordinates from the dataset
    coords = {dim: ds.coords[dim] for dim in dims if dim in ds.coords}
    
    # Create a DataArray with the given data, dimensions, and coordinates
    new_data_array = xr.DataArray(
        new_data,
        dims=dims,
        coords=coords
    )
    ds[var_name] = new_data_array

    return ds

## FileTracking is ft
ft = storms[[ 
    "INT_UTC_timeInit", "INT_UTC_timeEnd", "INT_duration", "INT_surfcumkm2_241K", "INT_velocityAvg", "INT_surfmaxkm2_241K", ## General characteristics
    "LC_lon", "LC_lat", "LC_UTC_time", "LC_ecc_241K", "LC_orientation_241K", "LC_surfkm2_241K", "LC_tb_90th", "LC_velocity" ## General characteristics
             ]]

ft_shape = tuple(ft.dims[dim] for dim in ft.dims)

LC_rcond_1mmh = np.full(ft_shape, np.nan)
LC_sigma_1mmh = np.full(ft_shape, np.nan)
INT_rcond_1mmh = np.full((ft_shape[0]), np.nan)
INT_sigma_1mmh = np.full((ft_shape[0]), np.nan)

LC_rcond_10mmh = np.full(ft_shape, np.nan)
LC_sigma_10mmh = np.full(ft_shape, np.nan)
INT_rcond_10mmh = np.full((ft_shape[0]), np.nan)
INT_sigma_10mmh = np.full((ft_shape[0]), np.nan)

LC_rcond_30mmh = np.full(ft_shape, np.nan)
LC_sigma_30mmh = np.full(ft_shape, np.nan)
INT_rcond_30mmh = np.full((ft_shape[0]), np.nan)
INT_sigma_30mmh = np.full((ft_shape[0]), np.nan)

to_drop = []
for iDCS, DCS_number in enumerate(ft.DCS_number.values): 
    print("Completion at ", 100*iDCS/len(ft.DCS_number.values), "% for DCS_number", DCS_number)
    start, end, lons, lats, speeds, times, time_smax, i_smax, lons_3d, lats_3d, speeds_3d, times_3d, speed_lon_3d, speed_lat_3d = st.get_frame_data(ft, DCS_number)
    # extents, slices_lon, slices_lat = st.get_extents_slices(lons_3d, lats_3d)
    times_3d_conv = [time for time in times_3d if time <= time_smax + st.settings["TIME_RANGE"][0]]
    t = len(times_3d_conv)
    if i_smax < 0: # or t<3
        continue #ft = ft.drop_sel(DCS_number = to_drop)
    else : 
        start, end, lons, lats, speeds, times, time_smax, i_smax, lons_3d, lats_3d, speeds_3d, times_3d, speed_lon_3d, speed_lat_3d = st.get_frame_data(ft, DCS_number)
        if i_smax + 1 < len(lons):
            extent, slice_lon, slice_lat = st.get_full_extent_slice(lons[:i_smax+1], lats[:i_smax+1], large_scale_frame_size=4)
        else:
            # If i_smax is the last index, use lons[:i_smax] and lats[:i_smax]
            extent, slice_lon, slice_lat = st.get_full_extent_slice(lons[:i_smax], lats[:i_smax], large_scale_frame_size=4)

        le_dico_long = {"latitude" : slice_lat, "longitude" : slice_lon}
        le_dico_court = {    "lat" : slice_lat,       "lon" : slice_lon}
        
        test = hdlr.load_seg(times[0], sel_dict = le_dico_long)[0].values #.sel(le_dico_long)

        t_smax = len(times[:i_smax+1])

        rcond_1mmh = np.zeros((t_smax))
        rcond_10mmh = np.zeros((t_smax))
        rcond_30mmh = np.zeros((t_smax))

        sigma_1mmh = np.zeros((t_smax))
        sigma_10mmh = np.zeros((t_smax))
        sigma_30mmh = np.zeros((t_smax))

        surf_seg_mask = [] 
        surf_rcond_1mmh = []
        surf_rcond_10mmh = []
        surf_rcond_30mmh = []

        all_rcond_1mmh = []
        all_rcond_10mmh = []
        all_rcond_30mmh = []

        for i in range(t_smax):
            age_to_smax = i/i_smax

            ##### CLOUD MASK ######
            seg = hdlr.load_seg(times[i], sel_dict = le_dico_long)[0].values #.sel(le_dico_long)
            seg_mask = np.full_like(seg, False, dtype = bool)
            seg_mask[seg == DCS_number] = True

            surf_seg_mask.append(np.sum(seg_mask))

            #### PREC ####
            prec = hdlr.load_var(cs, "Prec", times[i], sel_dict = le_dico_court).values
            prec[~seg_mask] = 0 # 1st timestep of seg_mask looks empty... 

            rcond_1mmh[i] = np.mean(prec[prec>1])
            rcond_10mmh[i] = np.mean(prec[prec>10])
            rcond_30mmh[i] = np.mean(prec[prec>30])
            
            all_rcond_1mmh.extend(prec[prec>1])
            all_rcond_10mmh.extend(prec[prec>10])
            all_rcond_30mmh.extend(prec[prec>30])

            sigma_1mmh[i] = np.sum(prec>1)/np.sum(seg_mask) if np.any(seg_mask) else 0 
            sigma_10mmh[i] = np.sum(prec>10)/np.sum(seg_mask) if np.any(seg_mask) else 0 
            sigma_30mmh[i] = np.sum(prec>30)/np.sum(seg_mask) if np.any(seg_mask) else 0 

            surf_rcond_1mmh.append(np.sum(prec>1))
            surf_rcond_10mmh.append(np.sum(prec>10))
            surf_rcond_30mmh.append(np.sum(prec>30))

    LC_rcond_1mmh[iDCS][start:end][:i_smax+1] = rcond_1mmh
    LC_rcond_10mmh[iDCS][start:end][:i_smax+1] = rcond_10mmh
    LC_rcond_30mmh[iDCS][start:end][:i_smax+1] = rcond_30mmh

    LC_sigma_1mmh[iDCS][start:end][:i_smax+1] = sigma_1mmh
    LC_sigma_10mmh[iDCS][start:end][:i_smax+1] = sigma_10mmh
    LC_sigma_30mmh[iDCS][start:end][:i_smax+1] = sigma_30mmh

    INT_rcond_1mmh[iDCS] = np.mean(all_rcond_1mmh)
    INT_rcond_10mmh[iDCS]= np.mean(all_rcond_10mmh)
    INT_rcond_30mmh[iDCS]= np.mean(all_rcond_30mmh)

    INT_sigma_1mmh[iDCS] = np.sum(surf_rcond_1mmh)/ np.sum(surf_seg_mask)
    INT_sigma_10mmh[iDCS]= np.sum(surf_rcond_10mmh)/ np.sum(surf_seg_mask)
    INT_sigma_30mmh[iDCS]= np.sum(surf_rcond_30mmh)/ np.sum(surf_seg_mask)


ft = add_variable_to_dataset(ft, LC_rcond_1mmh, 'LC_rcond_1mmh', ('DCS_number', 'time'))
ft = add_variable_to_dataset(ft, LC_rcond_10mmh, 'LC_rcond_10mmh', ('DCS_number', 'time'))
ft = add_variable_to_dataset(ft, LC_rcond_30mmh, 'LC_rcond_30mmh', ('DCS_number', 'time'))

ft = add_variable_to_dataset(ft, LC_sigma_1mmh, 'LC_sigma_1mmh', ('DCS_number', 'time'))
ft = add_variable_to_dataset(ft, LC_sigma_10mmh, 'LC_sigma_10mmh', ('DCS_number', 'time'))
ft = add_variable_to_dataset(ft, LC_sigma_30mmh, 'LC_sigma_30mmh', ('DCS_number', 'time'))

ft = add_variable_to_dataset(ft, INT_rcond_1mmh, 'INT_rcond_1mmh', ('DCS_number'))
ft = add_variable_to_dataset(ft, INT_rcond_10mmh, 'INT_rcond_10mmh', ('DCS_number'))
ft = add_variable_to_dataset(ft, INT_rcond_30mmh, 'INT_rcond_30mmh', ('DCS_number'))

ft = add_variable_to_dataset(ft, INT_sigma_1mmh, 'INT_sigma_1mmh', ('DCS_number'))
ft = add_variable_to_dataset(ft, INT_sigma_10mmh, 'INT_sigma_10mmh', ('DCS_number'))
ft = add_variable_to_dataset(ft, INT_sigma_30mmh, 'INT_sigma_30mmh', ('DCS_number'))


# Save the updated dataset to a new NetCDF file
filename = f"rcond_sigma_storms_dmin{duration_min}_smin{surfmaxkm2_min}_lat{region_latmin}_{region_latmax}_lon{region_lonmin}_{region_lonmax}.nc"
output_dir = os.path.join(st.settings["DIR_DATA_OUT"], cs.name)
output_path = os.path.join(output_dir, filename)

if os.path.exists(output_path):
    os.remove(output_path)
ft.to_netcdf(output_path)

# Print confirmation and go home :)
print(f"Dataset updated and saved to {output_path}. Time to head home!")
