import warnings
import time
import os 

import pickle
import copy
import gc
import logging

import numpy as np
import pandas as pd
import xarray as xr
import math

from datetime import datetime as dt

import matplotlib.pyplot as plt

class Grid(): 
    # except verbose i actually don't want any
    def __init__(self, casestudy, simpler_grid = True, fast = True, overwrite = True, verbose_steps = False, verbose=False):
        self.casestudy = casestudy
        self.settings = casestudy.settings
        
        ## Get the region borders
        self.n_lat = self.casestudy.lat_slice.stop - self.casestudy.lat_slice.start
        self.n_lon = self.casestudy.lon_slice.stop - self.casestudy.lon_slice.start
        # sub and add 0.5 makes it centered on middle of gridpoint. 
        self.lat_global = np.linspace(self.casestudy.lat_slice.start+0.5, self.casestudy.lat_slice.stop-0.5, self.n_lat) 
        self.lon_global = np.linspace(self.casestudy.lon_slice.start+0.5, self.casestudy.lon_slice.stop-0.5, self.n_lon) 
        ## Bool for running a quicker computation (can multiply by x2 or x8 depending on the function)
        self.fast = fast
        self.simpler_grid = simpler_grid

        ## explicitly verbose
        self.verbose = verbose

        ## very talkative one, mainly for computation time measurements
        self.verbose_steps = verbose_steps

        self.make_output_ready(overwrite)
        
        # Funcs to compute on variable 
        # Actually this should be done in CaseStudy and passed there, so that it'd be eazy to control which func for any var_id
        #### !!!!!!!!!!!!!!!!!!!!!!
        self.func_names = ['mean'] ## !!!!! usually ['max', 'mean'] especially for Prec !!!!! 
        self.cloud_vars = ["MCS_label", "MCS_Feng", "MCS_label_Tb_Feng", "Conv_MCS_label", 
                           "vDCS", "MCS_cond_Prec_15", "vDCS_cond_Prec_15", "clouds_cond_Prec_15", 
                           "sliding_MCS_cond_Prec_15", "sliding_clouds_cond_Prec_15",
                           "MCS_cond_Prec_25", "vDCS_cond_Prec_25", "clouds_cond_Prec_25", 
                           ]
        #### !!!!!!!!!!!!!!!!!!!!!!

        self.settings = casestudy.settings
        self.get_landmask()
        
    def _make_simple_grid(self):
        print("Making simpler grid")
        dir = self.settings['DIR_DATA_2D_IN']
        for file in os.listdir(dir):
            self.template_native_df = xr.open_dataset(os.path.join(dir,file)).sel(lon=self.casestudy.lon_slice,lat=self.casestudy.lat_slice)
            break
        
        if self.verbose_steps: print('compute coord centers...')
        self.lat_centers = self.template_native_df['lat'].sel(lat=self.casestudy.lat_slice).values
        self.lon_centers = self.template_native_df['lon'].sel(lon=self.casestudy.lon_slice).values
        
        if self.verbose_steps: print('compute pixel surface...')

        self.lat_length_on_center, self.lon_length_on_center = self._compute_length_centers_from_coord_borders()

        self.pixel_surface = self.lat_length_on_center * self.lon_length_on_center    

        if self.verbose_steps: print('compute native to global index slices')

        i_lat_native = []
        lat_min_global, lat_max_global = self.settings["BOX"][0], self.settings["BOX"][1]
        lat_global = list(range(lat_min_global, lat_max_global+1, 1)) #adapt the step 1 for finer or coarser in the futur
        self.lat = self.template_native_df.lat
        for lat_inf, lat_sup in zip(lat_global[:-1], lat_global[1:]):
            i_lat_native_for_global = np.where((self.lat >=lat_inf) & (self.lat <= lat_sup))[0]
            i_lat_native.append(i_lat_native_for_global)

        self.slices_i_lat = [slice(lat_native[0], lat_native[-1]+1) for lat_native in i_lat_native]

        i_lon_native = []
        lon_min_global, lon_max_global = self.settings["BOX"][2], self.settings["BOX"][3]
        lon_global = list(range(lon_min_global, lon_max_global+1, 1))
        self.lon = self.template_native_df.lon
        for lon_inf, lon_sup in zip(lon_global[:-1], lon_global[1:]):
            i_lon_native_for_global = np.where((self.lon >= lon_inf) & (self.lon <= lon_sup))[0]
            i_lon_native.append(i_lon_native_for_global)

        self.single_slice_j_lon = [slice(lon_native[0], lon_native[-1]+1) for lon_native in i_lon_native]
        self.slices_j_lon = [self.single_slice_j_lon for _ in self.slices_i_lat] #for more complex grid behaviour issues

        if self.verbose_steps: print("compute global pixel surface with native sum func and grid pixels")
        self.grid_surface = self.sum_data_from_center_to_global(self.pixel_surface)

    def make_output_ready(self, overwrite):
        filepath= os.path.join(self.casestudy.data_out, "grid_attributes.pkl")
        if not os.path.exists(filepath):
            self.overwrite = True
        else : self.overwrite = overwrite 

        if self.overwrite:
            if self.simpler_grid:
                self._make_simple_grid()
            else : 
                self._prepare_grid() 

            self.save_grid_attr(filepath)
        else :
            self.load_grid_attr(filepath)

    def save_grid_attr(self, filename):
        state = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def load_grid_attr(self, filename):
        print(f"Found grid attributes file , so loading {filename} instead of computing")
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        for attr, value in state.items():
            setattr(self, attr, value)

    def _prepare_grid(self):
        if  self.overwrite and self.fast : 
            # import one timestep to use its coordinates as a template
            dir = self.settings['DIR_DATA_2D_IN']
            for file in os.listdir(dir):
                self.template_native_df = xr.open_dataset(os.path.join(dir,file))
                break

            # compute
            
            if self.verbose_steps: print('-- Prepare comple and wrong grid')

            if self.verbose_steps: print('compute coord centers...')
            self.lat_centers = self.template_native_df['lat'].sel(lat=self.casestudy.lat_slice).values
            self.lon_centers = self.template_native_df['lon'].sel(lon=self.casestudy.lon_slice).values
            
            if self.verbose_steps: print('compute pixel surface...')

            self.lat_length_on_center, self.lon_length_on_center = self._compute_length_centers_from_coord_borders()

            self.pixel_surface = self.lat_length_on_center * self.lon_length_on_center      

            self.global_area = self.pixel_surface.sum()/self.n_lat/self.n_lon #depending on the remeshed grid point surface you want computed here
            self.global_lat_area = self.global_area*self.n_lon 

            self.lat_global = [i for i in range(self.n_lat)]
            self.lon_global = [j for j in range(self.n_lon)]

            ## We start by computing the area of each latitude band
            if self.verbose_steps: print('compute lat band area')
            self.lat_area = np.sum(self.pixel_surface, axis=1)
            self.cumsum_lat_area = np.cumsum(self.lat_area)

            if self.verbose_steps: print('compute i and alpha lat')
            self.i_min, self.i_max, self.alpha_i_min, self.alpha_i_max = self._get_i_and_alpha_lat()

            if not self.fast : 
                self.slices_i_lat = [slice(i_min+1, i_max) for i_min, i_max in zip(self.i_min[:,0], self.i_max[:,0])] 
            elif self.fast :
                self.slices_i_lat = [slice(i_min, i_max) for i_min, i_max in zip(self.i_min[:,0], self.i_max[:,0])]   
                 
            if self.verbose_steps: print('compute area by lon')
            self.area_by_lon_and_global_lat = self._compute_area_by_lon()
            self.cumsum_area_by_lon_and_global_lat = np.cumsum(self.area_by_lon_and_global_lat, axis = 1)

            if self.verbose_steps: print('compute j and alpha lon')
            self.j_min, self.j_max, self.alpha_j_min, self.alpha_j_max = self._get_j_and_alpha_lon()
            self.j_min, self.j_max =self.j_min.astype(int), self.j_max.astype(int)

            if self.verbose_steps: print('build slices j lon')
            self.slices_j_lon = self._build_slices_j_lon()

            if self.verbose_steps: print('compute grid surface')
            self.grid_surface = self.sum_data_from_center_to_global(self.pixel_surface)

            # remove template data
            # self.template_native_df = None  # USE GARBAGE COLLECTOR?
        
    def _compute_length_centers_from_coord_borders(self):
        lat_length = np.zeros(shape=(len(self.lat_centers), len(self.lon_centers)))
        lon_length = np.zeros(shape=(len(self.lat_centers), len(self.lon_centers)))
        
        self.lat_borders = self._get_coord_border_from_centers(self.lat_centers)
        self.lon_borders = self._get_coord_border_from_centers(self.lon_centers)
        
        for i_lat in range(len(self.lat_borders)-1):
            for j_lon in range(len(self.lon_borders)-1):
                lat1, lat2, lon1, lon2 = self.lat_borders[i_lat], self.lat_borders[i_lat+1], self.lon_borders[j_lon], self.lon_borders[j_lon+1]
                lat_length[i_lat, j_lon] = self.haversine(lat1, lon1, lat2, lon1)
                lon_length[i_lat, j_lon] = self.haversine(lat1, lon1, lat1, lon2)
        return lat_length, lon_length

    def _get_coord_border_from_centers(self, coord_centers):
        coord_borders = list()
        coord_borders.append(np.floor(coord_centers[0]))
        for i in range(len(coord_centers)-1):
            coord_borders.append((coord_centers[i]+coord_centers[i+1])/2)
        coord_borders.append(np.ceil(coord_centers[-1]))  
        return coord_borders
    
    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the distance between two points on the Earth (specified in decimal degrees)
        using the Haversine formula.
        """
        R = 6371  # Earth's radius in kilometers

        # Convert decimal degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c

        return distance
    
    def _get_i_and_alpha_lat(self):
        i_min, i_max = np.zeros((self.n_lat, self.n_lon)), np.zeros((self.n_lat, self.n_lon))
        alpha_i_min, alpha_i_max = np.ones((self.n_lat, self.n_lon)), np.ones((self.n_lat, self.n_lon))

        for i_lat, cum_length in enumerate(self.cumsum_lat_area):
            border_left = cum_length - self.lat_area[i_lat] ## does lat_area really needs to be a list or is a constant ? 
            border_right = cum_length
            
            for i in range(self.n_lat):
                cum_global_length = (i+1)*self.global_lat_area
                
                if cum_global_length > border_left and ((cum_global_length < border_right) or (math.isclose(cum_global_length, border_right))): # here we must use math.isclose for the behavior of <=
                    bottom_contrib = (cum_global_length - border_left)/self.lat_area[i_lat]
                    top_contrib = (border_right - cum_global_length)/self.lat_area[i_lat]           
                    # if self.verbose : print('local', i_lat, cum_length,'global',  i, cum_global_length,'borders',  border_left, border_right, 'contribs', bottom_contrib, top_contrib)
                    
                    if not self.fast:
                        if i != self.n_lat-1:
                            i_min[i+1, :] = i_lat
                            alpha_i_min[i+1, :] = top_contrib if not (math.isclose(cum_global_length, border_right)) else 0
                        

                        i_max[i, :] = i_lat
                        alpha_i_max[i, :] = bottom_contrib if not (math.isclose(cum_global_length, border_right)) else 1
                    
                    if self.fast:
                        if bottom_contrib > top_contrib : ## hence this area deserves to be kept in the regridded new area
                            if i != self.n_lat-1:
                                i_min[i+1, :] = i_lat 
                            i_max[i,:] = i_lat
                            
                        elif top_contrib > bottom_contrib : ##there this area must be left to the nex regridded new area so we update our index to i_lat-1 , the previous area index
                            if i != self.n_lat-1:
                                i_min[i+1, :] = i_lat-1
                            i_max[i, :] = i_lat-1
                    
        return i_min.astype(int), i_max.astype(int), alpha_i_min, alpha_i_max
    
    def _compute_area_by_lon(self):
            area_by_lon = np.zeros((self.n_lat, self.lon_centers.shape[0]))
            for j_lon in range(self.lon_centers.shape[0]):
                for i, slice_i_lat in enumerate(self.slices_i_lat):
                    # if not self.fast : 
                    i_min = self.i_min[i, :]
                    i_min = self._check_all_values_same(i_min)
                    i_max = self.i_max[i, :]
                    i_max = self._check_all_values_same(i_max)
                    alpha_i_min = self.alpha_i_min[i, :]
                    alpha_i_min = self._check_all_values_same(alpha_i_min)
                    alpha_i_max = self.alpha_i_max[i, :]
                    alpha_i_max = self._check_all_values_same(alpha_i_max)
                
                    ## print i_min, i_max, alpha_i_min, alpha_i_max
                    #if self.verbose : print(i, i_min, i_max, alpha_i_min, alpha_i_max)
                    bottom_sum = self.pixel_surface[i_min,j_lon]*alpha_i_min
                    #if self.verbose : print(bottom_sum)
                    top_sum = self.pixel_surface[i_max,j_lon]*alpha_i_max
                    #if self.verbose : print(top_sum)
                    mid_sum = np.sum(self.pixel_surface[slice_i_lat, j_lon])
                    #if self.verbose : print(mid_sum)
                    area_by_lon[i, j_lon] = mid_sum+bottom_sum+top_sum
                    #print everything 
                    # elif self.fast :
                    #     area_by_lon[i, j_lon] = np.sum(self.pixel_surface[slice_i_lat, j_lon])
                    if False : print('i', i, 'j_lon', j_lon, 'i_min', i_min, 'i_max', i_max, 'slice_i_lat', slice_i_lat, 'alpha_i_min', alpha_i_min, 'alpha_i_max', alpha_i_max, 'bottom_sum', bottom_sum, 'top_sum', top_sum, 'mid_sum', mid_sum, 'area_by_lon', area_by_lon[i, j_lon])
            
            return area_by_lon
    
    def _get_j_and_alpha_lon(self):
        j_min, j_max = np.zeros((self.n_lat, self.n_lon)), np.zeros((self.n_lat, self.n_lon))
        alpha_j_min, alpha_j_max = np.ones((self.n_lat, self.n_lon)), np.ones((self.n_lat, self.n_lon))
        for i in range(self.n_lat):
            cumsum_area_by_lon = self.cumsum_area_by_lon_and_global_lat[i, :]
            for j_lon, cum_length in enumerate(cumsum_area_by_lon):
                border_left = cum_length - self.area_by_lon_and_global_lat[i, j_lon]
                border_right = cum_length
                
                for j in range(self.n_lon):
                    cum_global_length = (j+1)*self.global_area
                    
                    if cum_global_length > border_left  and ((cum_global_length) < border_right or (math.isclose(cum_global_length, border_right))):
                        left_contrib = (cum_global_length - border_left)/self.area_by_lon_and_global_lat[i, j_lon]
                        right_contrib = (border_right - cum_global_length)/self.area_by_lon_and_global_lat[i, j_lon]
                        
                        # if not self.fast:
                            # if self.verbose : print('local', j_lon, cum_length,'global',  j, cum_global_length,'borders',  border_left, border_right, 'contribs', left_contrib, right_contrib)
                        if j!= self.n_lon-1:
                            j_min[i, j+1] = j_lon
                            alpha_j_min[i, j+1] = right_contrib if not (math.isclose(cum_global_length, border_right)) else 0
                            
                        j_max[i, j] = j_lon
                        alpha_j_max[i, j] = left_contrib if not (math.isclose(cum_global_length, border_right)) else 1
                        
                        # elif self.fast: 
                        #     if left_contrib > right_contrib:
                        #         if j != self.n_lon-1:
                        #             j_min[i, j+1] = j_lon
                        #         j_max[i,j] = j_lon
                        #     elif right_contrib > left_contrib:
                        #         if j!= self.n_lon-1:
                        #             j_min[i, j+1] = j_lon
                        #         j_max[i,j] = j_lon
                                
        return j_min, j_max, alpha_j_min, alpha_j_max   
    
    def _build_slices_j_lon(self):
        slices_j_lon = np.empty((self.n_lat, self.n_lon), dtype=object)
        for i in range(self.n_lat):
            for j in range(self.n_lon):
                if not self.fast :  
                    slices_j_lon[i, j] = slice(int(self.j_min[i, j])+1, int(self.j_max[i, j])) 
                elif self.fast :
                    slices_j_lon[i, j] = slice(int(self.j_min[i, j])  , int(self.j_max[i, j])) 
        return slices_j_lon
    
    def sum_data_from_center_to_global(self, data_on_center):
        x = data_on_center
        X = np.zeros((self.n_lat, self.n_lon))
        for i, slice_i_lat in enumerate(self.slices_i_lat):
            for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                if self.verbose : pass #print(slice_i_lat, slice_j_lon)
                if self.fast : 
                    if len(np.shape(x==2)):
                        X[i,j] = np.nansum(x[slice_i_lat, slice_j_lon]) # passed from sum to nansum 01/15/2024
                    else : # assume x is 3 shape t,y,x
                        X[i,j] = np.nansum(x[:, slice_i_lat, slice_j_lon])

                else : 
                    mid_sum = np.sum(x[slice_i_lat, slice_j_lon])
                    bottom_sum = np.sum( x[self.i_min[i,j], slice_j_lon]*self.alpha_i_min[i,j])
                    top_sum = np.sum( x[self.i_max[i,j], slice_j_lon]*self.alpha_i_max[i,j])
                    left_sum = np.sum( x[slice_i_lat, self.j_min[i,j]]*self.alpha_j_min[i,j])
                    right_sum = np.sum( x[slice_i_lat, self.j_max[i,j]]*self.alpha_j_max[i,j])
                    bottom_left_corner = x[self.i_min[i,j], self.j_min[i,j]]*self.alpha_j_min[i,j]*self.alpha_i_min[i,j]
                    bottom_right_corner = x[self.i_min[i,j], self.j_max[i,j]]*self.alpha_j_max[i,j]*self.alpha_i_min[i,j]
                    top_left_corner = x[self.i_max[i,j], self.j_min[i,j]]*self.alpha_j_min[i,j]*self.alpha_i_max[i,j]
                    top_right_corner = x[self.i_max[i,j], self.j_max[i,j]]*self.alpha_j_max[i,j]*self.alpha_i_max[i,j]
                    X[i, j] = mid_sum+bottom_sum+top_sum+left_sum+right_sum+bottom_left_corner+bottom_right_corner+top_left_corner+top_right_corner
        del x
        gc.collect()
        return X
    
    def _check_all_values_same(self, arr):
        first_value = arr[0]
        for value in arr:
            if value != first_value:
                raise ValueError("Array contains different values")

        return first_value
    
    def plot_grid(self, w = 10, h = 5):
        # Create a 1x2 grid of subplots
        plt.figure(figsize=(w, h))  # Adjust the figure size as needed
        plt.subplot(1, 2, 1)  # Create the first subplot

        ## Here checks if the global_pixel_surf in ds looks correct (and is accessible so that grid.ds is not corrupted)
        X = self.grid_surface
        plt.imshow(X, origin='lower')
        plt.colorbar()
        plt.title("Grid Pixel Surface")

        plt.subplot(1, 2, 2)  # Create the second subplot
        X = self.pixel_surface
        plt.imshow(X, origin='lower')
        plt.colorbar()
        plt.title("Pixel Surface")

        plt.tight_layout()  # Ensures that subplots do not overlap

        plt.show()  # Display the entire figure with both subplots

    def get_var_ds_file(self, var_id):
        filename = var_id.lower()+'.nc'
        file = os.path.join(self.casestudy.data_out, filename)
        return(file)

    def get_var_id_ds(self, var_id):
        file = self.get_var_ds_file(var_id)
        if not os.path.exists(file):
            print(f"\n Woah,\n the netcdf for this variable {var_id} didn't exist yet, let's make it from scratch : \n")
            dims_global = ['lat_global', 'lon_global', 'days']
            days = list(self.casestudy.days_i_t_per_var_id[var_id].keys())
            coords_global = {'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': days}
            da_global = xr.DataArray(None, dims = dims_global, coords = coords_global) # should add pixel surf here (extended by correct number of days)
            ds = xr.Dataset({'global_pixel_surf': da_global}) 
            ds.to_netcdf(file)    
        else:
            ds = xr.open_dataset(file, engine = "netcdf4")
            # Check if the loaded dataset contains the required coordinates
            required_coordinates = ['lat_global', 'lon_global', 'days']
            missing_coordinates = [coord for coord in required_coordinates if coord not in ds.coords]
            if missing_coordinates:
                print(f"The dataset is missing the following coordinates: {missing_coordinates}")
        return ds

    def get_landmask(self):
        # file = self.get_var_ds_file("LANDMASK")
        file = "/scratchx/mcarenso/tempest/landmask.nc" ## try to bypass to adapt to other sim where no corresponding landmask is available
        if not os.path.exists(file):
            print("Creating Earth (rough borders)")
            #arbitratry choice here, corresponds to i_t = 960
            filepath_var = os.path.join(self.settings["DIR_DATA_2D_IN"], "DYAMOND_9216x4608x74_7.5s_4km_4608_0000230640.LANDMASK.2D.nc")
            landmask = xr.open_dataarray(filepath_var).load().sel(lon=self.casestudy.lon_slice,lat=self.casestudy.lat_slice)[0]
            landmask_regridded = np.zeros((self.n_lat, self.n_lon))
            for i, slice_i_lat in enumerate(self.slices_i_lat):
                for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                    lm = landmask[slice_i_lat, slice_j_lon]
                    lm_unique = np.unique(lm)
                    if 1.0000314 in lm_unique :
                        landmask_regridded[i,j] = 1
                    else : 
                        landmask_regridded[i,j] = 0
                        
            dims_global = ['lat_global', "lon_global"]
            coords_global = {'lat_global' : self.lat_global, 'lon_global' : self.lon_global}
            da_global = xr.DataArray(landmask_regridded, dims = dims_global, coords = coords_global)
            ds = xr.Dataset({'Landmask' : da_global})
            ds.to_netcdf(file)
        else : 
            ds = xr.open_dataset(file)
            required_coordinates = ['lat_global', 'lon_global']
            missing_coordinates = [coord for coord in required_coordinates if coord not in ds.coords]
            if missing_coordinates:
                # Il est corrompu on dirait...
                print(f"The dataset is missing the following coordinates: {missing_coordinates}")
        self.mask_ocean  = ds.sel(lat_global=self.casestudy.lat_slice).Landmask == 0 ## this fucker is not compatible with -180, 180...
        self.mask_ocean = self.mask_ocean.values[:,:,np.newaxis]
        self.mask_land = ~self.mask_ocean
        self.mask_all = np.logical_or(self.mask_ocean, self.mask_land)
        return ds
                
# These are super unclear but I got used to it..
    def compute_funcs_for_var_id(self, var_id='Prec', overwrite_var_id = False):
        """
        Save to netcdf all the funcs to apply to var_id, that were not already a key. 
        """
        var_ds = self.get_var_id_ds(var_id)         
        funcs = []
        keys = []
        keys_loaded = [key for key in list(var_ds.variables) if var_id in key] 
        
        if var_id == "Prec" or "Prec_lowRes":
            ## this should desactivates everything but not sure, (it's due to the fact that the second time i coded it super well :) )
            funcs = ["cond_alpha_00", "cond_alpha_01", "cond_alpha_05", "cond_alpha_10","cond_alpha_20",  "cond_alpha_25", "cond_alpha_33", "cond_alpha_40",
                     "cond_alpha_50", "cond_alpha_55","cond_alpha_60", "cond_alpha_62", "cond_alpha_64", "cond_alpha_65", "cond_alpha_66", "cond_alpha_67", 
                     "cond_alpha_68", "cond_alpha_69", "cond_alpha_70", "cond_alpha_71", "cond_alpha_72", "cond_alpha_73", "cond_alpha_74", "cond_alpha_75",
                     "cond_alpha_76", "cond_alpha_77", "cond_alpha_78", "cond_alpha_79", "cond_alpha_80", "cond_alpha_81", "cond_alpha_82", "cond_alpha_83", 
                     "cond_alpha_84", "cond_alpha_85", "cond_alpha_86", "cond_alpha_87", "cond_alpha_88", "cond_alpha_89", "cond_alpha_90", "cond_alpha_91",
                     "cond_alpha_92", "cond_alpha_93", "cond_alpha_94",  "cond_alpha_95", "cond_alpha_96", "cond_alpha_97", "cond_alpha_98", "cond_alpha_99",
                     "convective_01", "convective_02", "convective_03", "convective_04", "convective_05", "convective_06", "convective_08", "convective_10", 
                     "convective_12", "convective_15", "convective_20","convective_25", "convective_30", "convective_40", "convective_50", "max"] 
                     ### CAREFULL HERE, max mean (or single_timestep_regridding funcs must always be behind daily ones)
            # old_funcs= ["heavy", "supra", "ultra", "wet", "convective"]
        elif var_id == "vDCS_Conv_Prec" or var_id == "vDCS_Strat_Prec":
            funcs = ["daily_weighted_mean"]
        else: 
            funcs = self.func_names
        
        ## funcs are sent at the level where data is loaded, so must check cleanly which are required
        funcs_to_compute = []
        for func_name in funcs: # for now func_names is the same for all var_id, but it should be updated in a json to be in the same fashion that days_i_t_per_var_id
            key = f"{func_name}_{var_id}"
            if key in keys_loaded and not overwrite_var_id: 
                if not overwrite_var_id : print('%s already computed, skipping...'%key)
            else :
                if overwrite_var_id : print(f"compute {keys_loaded} again because of overwrite parameter")
                funcs_to_compute.append(func_name) 
                keys.append(key) 
                            
                if 'heavy' == func_name: # Actually func heavy returns 4 keys
                    keys += ["Alpha_95", "Sigma_95", "bis_mean_Prec"]
                if "supra" == func_name:
                    keys += ["Alpha_99", "Sigma_99"]
                if "ultra" == func_name:
                    keys+= ["Alpha_99_99", "Sigma_99_99"]
                if "wet" == func_name:
                    keys+= ["Alpha_1mm_per_h", "Sigma_1mm_per_h"]
                if "convective" == func_name:
                    keys+= ["Alpha_99_99_native", "Sigma_99_99_native"]
                if "convective" in func_name :
                    keys+=["Alpha_"+key, "Sigma_"+key]
                if "cond_alpha" in func_name:
                    keys+=["mean_unweighted_"+var_id, "Sigma_"+key, "threshold_"+key, "Sigma_intra_day_"+key]
        
        # Overide - If var_id is MCS (or maybe later contains MCS) - the funcs and keys to only compute the MCS_label
        if var_id in self.cloud_vars: 
            key = var_id
            keys = [key, "Rel_surface"]
            funcs_to_compute = [None, None] # adding None here, adds funcs to MCS regridding

        if len(keys)>0 : 
            print(f"These keys : {keys} have to be computed.")
            da_days_funcs = [[] for _ in funcs_to_compute]
            for func in funcs_to_compute : 
                if func is not None: #MCS overide issue
                    if "cond_alpha" in func: 
                        da_days_funcs += [[], [], [], []]
                    elif "convective" in func:
                        da_days_funcs += [[], []]
                    elif "supra" == func or "ultra" == func or "wet" == func or "convective"==func or "heavy" == func:
                        da_days_funcs += [[], []]
            
            days = list(self.casestudy.days_i_t_per_var_id[var_id].keys())
            for day in days:
                print(f"\n computing day {day} for funcs {funcs_to_compute}")
                da_funcs = self.regrid_funcs_and_save_by_day(day, var_id, funcs_to_compute)
                # if heavy is in funcs_to_compute, da_funcs will naturally have 4 items as if 3 more funcs but it's just 4 different keys
                for i_f, da_func in enumerate(da_funcs):
                    da_days_funcs[i_f].append(da_func)
                del da_funcs
                del da_func
                gc.collect()

            for da_all_days, key in zip(da_days_funcs, keys) : 
                if self.verbose : print("concat da days before saving")
            ## concat the list of dataarrays along days dimensions
                da_var_regrid = xr.concat(da_all_days, dim = 'days').sortby('days')
                ## By doing assign we actually keep the already existing variables of the netcdf
                ## If we were adding specific days to already existing key, we should do it an other way.
                ## Could use the to_complete bool mentionned earlier in the function
                var_ds = var_ds.assign(**{key: da_var_regrid})
                
            del da_all_days
            del da_days_funcs
            del da_var_regrid
            gc.collect()
            
            file = self.get_var_ds_file(var_id)
            os.remove(file)
            var_ds.to_netcdf(file) ## this forces the update of the stored .nc
            var_ds.close()
        
        else : print("nothing to compute then")

    def regrid_funcs_and_save_by_day(self, day, var_id='Prec', funcs=['max', 'mean']):
        """
        Compute multiple functions on new grid for a given day
        """	
        outputs_da = [] ##list of full day DataArray for each func (so typically of len 1 for MCS_label, 2 for any variable, 3 for Prec)
        var_regridded_per_funcs = self.regrid_funcs_for_day(day, var_id=var_id, funcs_to_compute=funcs)

        for var_regridded in  var_regridded_per_funcs:
            if var_id in self.cloud_vars:  
                n_MCS = var_regridded.shape[3] # catch correct dimension here for labels_yxtm 
                da_day = xr.DataArray(var_regridded, dims=['lat_global', 'lon_global', 'days', 'MCS'], 
                                        coords={'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': [day], 'MCS':np.arange(n_MCS)})
                
            else :
                if isinstance(var_regridded, list): #why when what (random guess it's for 3d extracted variables, lets skip)
                    da_day = xr.DataArray(var_regridded[0], dims=['lat_global', 'lon_global', 'days'], 
                                            coords={'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': [day]})
                else : 
                    if len(np.shape(var_regridded))==3:
                        if self.verbose : print("got a 3 shape da")
                        da_day = xr.DataArray(var_regridded   , dims=['lat_global', 'lon_global', 'days'], 
                                                coords={'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': [day]})
                    elif len(np.shape(var_regridded))==4:
                        if self.verbose : print("got a 4 shape da")
                        da_day = xr.DataArray(var_regridded, dims = ['lat_global', 'lon_global', 'time', 'days'],
                                              coords={'lat_global' : self.lat_global, 'lon_global' : self.lon_global, 'time' : np.arange(48), 'days': [day]})
            outputs_da.append(da_day)
        del var_regridded_per_funcs
        del da_day
        gc.collect()
        return outputs_da

    def regrid_funcs_for_day(self, day, var_id='Prec', funcs_to_compute=['max', 'mean']):
        """
        Compute multiple functions on new grid for a given day and return the results as a list.
        Simultaneously works for either : 
        - convective+cond_alpha only funcs
        - MCS vars func 
        - classic functions out of previous specifications

        Args:
            day (str): The specific day for which regridding will be performed.
            var_id (str, optional): A variable identifier that determines what kind of data is being regridded.
            funcs_to_compute (list, optional): A list of functions to be applied to the data during regridding.

        Returns:
            list: A list of regridded data for each function in funcs_to_compute, in the same order.

        """
        temp_funcs_to_compute = copy.deepcopy(funcs_to_compute)        
        threshold_alpha = []  # Initialize as an empty list
        threshold_convective = []  # Initialize as an empty list
        alpha_cond_bool = False
        convective_cond_bool = False

        for temp_func in temp_funcs_to_compute[:]:  # Iterate over a copy to safely remove items
            if temp_func is not None:  # same MCS override problem
                if "cond_alpha" in temp_func:
                    threshold_alpha.append(float("0." + temp_func[-2:]))  # Append to list
                    alpha_cond_bool = True
                    temp_funcs_to_compute.remove(temp_func)
                    
                elif "convective" in temp_func:
                    threshold_convective.append(float(temp_func[-2:]))  # Append to list
                    convective_cond_bool = True
                    temp_funcs_to_compute.remove(temp_func)

        if not threshold_alpha: alpha_cond_bool = False
        if not threshold_convective: convective_cond_bool = False

        if alpha_cond_bool or convective_cond_bool:
            print("loading whole day data for day", day, "for alpha_cond")
            assert var_id == 'Prec' or var_id == "Prec_lowRes"
            var_day = []
            i_t_for_day = self.casestudy.days_i_t_per_var_id[var_id][day]
            i_t_within_day = np.array(i_t_for_day) % 48
            for i_t in i_t_for_day:
                var_day.append(self.casestudy.handler.load_var(self, var_id, i_t))
            var_day = xr.concat(var_day, dim='time')  # could add coord i_t here but im too dumb with dataArrays coords
            day_per_diag = []
            
            if alpha_cond_bool:
                for threshold in threshold_alpha:
                    print("compute alpha_cond for threshold", threshold)
                    day_per_diag += self.get_cumsum_data_from_center_to_global(var_day, i_t_within_day, threshold)
                    
            if convective_cond_bool:
                for threshold in threshold_convective:
                    print("compute convective_cond for threshold", threshold)
                    day_per_diag += self.get_tail_data_from_center_to_global(var_day, threshold, False)
                    
            if 'day_per_func' in locals() or 'day_per_func' in globals():
                day_per_func += day_per_diag
            else: 
                day_per_func = day_per_diag

        def regrid_single_time_step(i_t, var_id, temp_funcs_to_compute):
            """
            Regrid data for a single time step.

            Args:
                idx (int): Index representing the time step.
                var_id (str): A variable identifier.

            Returns:
                list: A list of regridded data for each function in funcs_to_compute.
            """
            # Get the data for the day
            var_current = self.casestudy.handler.load_var(self, var_id, i_t)

            # Compute each func(var) for the current time step
            results = []
            for func in temp_funcs_to_compute:
                if var_current is not None:
                    var_regrid_idx = getattr(self, 'spatial_%s_data_from_center_to_global' % func)(var_current)
                else:
                    # Create an xarray filled with nans
                    var_regrid_idx = self.create_empty_array()
                    print('Regridding NaNs')

                results.append(var_regrid_idx)

            del var_current
            del var_regrid_idx
            gc.collect()

            return results
        
        if var_id in self.cloud_vars:
            ## MCS have a special treatment as they are the storm tracking inputs, they don't use regrid_single_time_step
            ## Any variable with MCS within should actually be treated differently.
            
            var_day = []
            for i_t in self.casestudy.days_i_t_per_var_id[var_id][day]:
                print(f"Loading {i_t} for day {day}")
                var_day.append(self.casestudy.handler.load_var(self, var_id, i_t)) # This loads quite a lot into memory. and where chunks couldcome usefull
            var_day = xr.concat(var_day,dim='time')
            
            labels_regrid, mcs_rel_surface = self.get_labels_data_from_center_to_global(var_day)
            labels_regrid = np.expand_dims(labels_regrid, axis=2) # try to add the dim for day on second position so that MCS is on third for labels_yxtm
            mcs_rel_surface = np.expand_dims(mcs_rel_surface, axis=2)
            del var_day 
            gc.collect()

            return [labels_regrid, mcs_rel_surface] # we put it into a list so that it is the same fashion than over variables that could have multiple funcs

        elif "daily_weighted_mean" in temp_funcs_to_compute:
            print("loading whole day data for day", day, "for daily weighted mean")
            assert var_id == 'vDCS_Conv_Prec' or var_id == "vDCS_Strat_Prec"
            var_day = []
            i_t_for_day = self.casestudy.days_i_t_per_var_id[var_id][day]

            for i_t in i_t_for_day:
                var_day.append(self.casestudy.handler.load_var(self, var_id, i_t))
            print("finished loading ", day)
            var_day = xr.concat(var_day, dim='time')
            
            day_per_diag = []
            
            day_per_diag += self.daily_spatial_mean_weighted(var_day)

            if 'day_per_func' in locals() or 'day_per_func' in globals():
                day_per_func += day_per_diag
            else: 
                day_per_func = day_per_diag

        elif len(temp_funcs_to_compute)>0 :  
        ## here we treat the funcs that uses regird_single_time_step. 
        # this means the result must then be aggregated from "hourly" to daily with a native numpy operation like currently np.nanmax / np.nanmean
        # Otherwise you'll have to load whole data
        # The fact that this section occurs after the cond_alhpa, and convective ones means that the passed func for these operations must be behind the ones for cond and conv..
        # Since output are not managed as a dict (because multiple diag per func/key) order must be preserved   
            # Loop over i_t, then loop over funcs_to_compute as to call regrid_single_time_step only once per i_t
            all_i_t_for_day_per_func = [[] for _ in temp_funcs_to_compute]
            for i_t in self.casestudy.days_i_t_per_var_id[var_id][day]:
                # Regrid time step for each func
                results = regrid_single_time_step(i_t, var_id, temp_funcs_to_compute)
                for i_f, result in enumerate(results):
                    all_i_t_for_day_per_func[i_f].append(result)
                del results
                gc.collect()

            # Stack and aggregate the data for each func
            # This stacking-aggregation is dependent of funcs to compute,
            # don't forget to modify this step if you're adding a new function 
            # or to include it in the method/function... at day level then.
            for i_f, func in enumerate(temp_funcs_to_compute):
                stacked_array = np.stack(all_i_t_for_day_per_func[i_f], axis=0)
                aggregated_array = getattr(np, 'nan%s' % func)(stacked_array, axis=0)
                day_for_func = [np.expand_dims(aggregated_array, axis = -1)] # make it a list
            
                if 'day_per_func' in locals() or 'day_per_func' in globals():
                    day_per_func += day_for_func
                else: 
                    day_per_func = day_for_func
                #maybe i'm overdoing it there
                del stacked_array
                del aggregated_array
                del day_for_func
                gc.collect()

        return day_per_func

### Add-ons methods for special regridding, should be simplified to help users

    def spatial_mean_data_from_center_to_global(self, data_on_center):
        """
        Returns the mean of data_on_center, weighted by the relative value of the initial pixel divided by the 
        final grid_surface. 
        """
        weights = self.pixel_surface
        if np.shape(data_on_center)!=np.shape(weights): ## this is specifically for sst
            data_on_center = data_on_center.values.reshape(data_on_center.shape[0]//2, 2, data_on_center.shape[1]//2, 2).mean(axis=(1, 3))
        x = data_on_center*weights if type(data_on_center) == np.ndarray else data_on_center.values*weights
        X = self.sum_data_from_center_to_global(data_on_center = x)
        global_weights = self.grid_surface
        # return np.expand_dims(X/global_weights , axis =-1)
        return X/global_weights

    def daily_spatial_mean_weighted(self, data_on_center):
        """
        Returns the mean of data_on_center, weighted by the relative value of the initial pixel divided by the 
        final grid_surface. 
        """
        data_shape = data_on_center.shape #3rd dim can vary due to time
        # weights = np.repeat(np.expand_dims(self.pixel_surface, axis=0), data_shape[0], axis=0)
        x = data_on_center if type(data_on_center) == np.ndarray else data_on_center.values #*weights ; *weights
        X = self.nanmean_data_from_center_to_global(data_on_center = x)
        return [X] #/ self.grid_surface

    def nanmean_data_from_center_to_global(self, data_on_center):
        x = data_on_center
        X = np.zeros((self.n_lat, self.n_lon))
        for i, slice_i_lat in enumerate(self.slices_i_lat):
            for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                if self.fast : 
                    if len(np.shape(x))==2:
                        X[i,j] = np.nanmean(x[slice_i_lat, slice_j_lon]) # passed from sum to nansum 01/15/2024
                    else : # assume x is 3 shape t,y,x
                        X[i,j] = np.nanmean(x[:, slice_i_lat, slice_j_lon])
        del x
        gc.collect()
        return np.expand_dims(X, axis = 2)

    def spatial_max_data_from_center_to_global(self, data_on_center):
        x = data_on_center if type(data_on_center) == np.ndarray else data_on_center.values
        X = np.zeros((self.n_lat, self.n_lon))
        if not self.fast : alpha_max = self.__build_alpha_max__()
        for i, slice_i_lat in enumerate(self.slices_i_lat):
            for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                if not self.fast :
                    m = np.nanmax(x[slice_i_lat, slice_j_lon].flatten())
                    b = np.nanmax((x[self.i_min[i,j], slice_j_lon]*alpha_max[self.i_min[i,j], slice_j_lon]).flatten())
                    t = np.nanmax((x[self.i_max[i,j], slice_j_lon]*self.alpha_max[self.i_max[i,j], slice_j_lon]).flatten())
                    l = np.nanmax((x[slice_i_lat, self.j_min[i,j]]*self.alpha_max[slice_i_lat, self.j_min[i,j]]).flatten())
                    r = np.nanmax((x[slice_i_lat, self.j_max[i,j]]*self.alpha_max[slice_i_lat, self.j_max[i,j]]).flatten())
                    blc = (x[self.i_min[i,j], self.j_min[i,j]]*self.alpha_max[self.i_min[i,j], self.j_min[i,j]]).flatten()
                    btc = (x[self.i_min[i,j], self.j_max[i,j]]*self.alpha_max[self.i_min[i,j], self.j_max[i,j]]).flatten()
                    tlc = (x[self.i_max[i,j], self.j_min[i,j]]*self.alpha_max[self.i_max[i,j], self.j_min[i,j]]).flatten()
                    trc = (x[self.i_max[i,j], self.j_max[i,j]]*self.alpha_max[self.i_max[i,j], self.j_max[i,j]]).flatten()
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        X[i, j] = np.nanmax(np.array([m, b, t, l, r, blc, btc, tlc, trc],dtype=object))
                
                elif self.fast :
                    X[i,j] = np.nanmax(x[slice_i_lat, slice_j_lon].flatten())
        # print(X.shape, X)
        del x 
        gc.collect()
        return X    

    def get_labels_data_from_center_to_global(self, data_on_center):
        """From segmentation mask, store 1 value of each label appearing at each location in new grid.
        Input: daily-concatenated variable"""

        n_MCS = 30 # new dimension size to store MCS labels
        
        x = data_on_center
        X = np.full((self.n_lat, self.n_lon,n_MCS),np.nan)
        X_rel_surface = np.full((self.n_lat, self.n_lon,n_MCS),np.nan)
        
        for i, slice_i_lat in enumerate(self.slices_i_lat): 
            if i%10 == 0: print(i,end='..')
            for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                x_subsets = x[:,slice_i_lat, slice_j_lon] # La 1ère dimension est celle du temps (i_t)   
                arr = x_subsets.values
                unique, unique_counts = np.unique(arr[~np.isnan(arr)].astype(int),return_counts=True)
                n_labs = len(unique)
                X[i, j,:n_labs] = unique
                surf = np.prod(np.array(np.shape(x_subsets)))
                X_rel_surface[i,j,:n_labs] = unique_counts/surf
        return X, X_rel_surface
    
    def get_tail_data_from_center_to_global(self, data_on_center, threshold, is_relative):
        x = data_on_center if type(data_on_center) == np.ndarray else data_on_center.values #1st axis becomes time
        mean_check = np.full((self.n_lat, self.n_lon), np.nan)
        Alpha = np.full((self.n_lat, self.n_lon), np.nan)
        Sigma = np.full((self.n_lat, self.n_lon), np.nan)
        Rcond = np.full((self.n_lat, self.n_lon), np.nan)
        for i, slice_i_lat in enumerate(self.slices_i_lat):
            if i%10==0 : print(i, end='..')
            for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                x_subset = np.sort(x[:, slice_i_lat, slice_j_lon].flatten()) 
                perc = np.percentile(x_subset, threshold) if is_relative else threshold
                mean = np.mean(x_subset)
                rcond = np.mean(x_subset[x_subset>perc])
                sigma = np.sum(x_subset>perc)/len(x_subset)
                alpha = (rcond*sigma)/mean
                
                mean_check[i,j] = mean
                Alpha[i,j] = alpha
                Sigma[i,j] = sigma
                Rcond[i,j] = rcond

        if threshold == 95 : 
            output = [np.expand_dims(Rcond, axis =-1), # Is called heavy_Prec afterward because of how key works
                    np.expand_dims(Alpha, axis = -1), 
                    np.expand_dims(Sigma, axis =-1),
                    np.expand_dims(mean_check, axis = -1)]
        else:
            output= [np.expand_dims(Rcond, axis =-1), # Is called heavy_Prec afterward because of how key works
                np.expand_dims(Alpha, axis = -1), 
                np.expand_dims(Sigma, axis =-1)]        
        
        # expand so that we can concatenate futur dataarrays along day dim
        return output

    def get_cumsum_data_from_center_to_global(self, data_on_center, i_t_within_day, threshold = 0.85):
        x = data_on_center if type(data_on_center) == np.ndarray else data_on_center.values #1st axis becomes time
        mean_check = np.full((self.n_lat, self.n_lon), np.nan)
        sigma_global = np.full((self.n_lat, self.n_lon), np.nan)
        rate_cond = np.full((self.n_lat, self.n_lon), np.nan)
        precip_cond = np.full((self.n_lat, self.n_lon), np.nan)
        sigma_global_time = np.full((self.n_lat, self.n_lon, 48), np.nan) # at most 48 timestepwithin day 
        
        for i, slice_i_lat in enumerate(self.slices_i_lat):
            # if i%10==0 : print(i, end='..')
            for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                x_subset_dirty = np.sort(x[:, slice_i_lat, slice_j_lon].flatten()) 
                x_subset = x_subset_dirty[~np.isnan(x_subset_dirty)]
                x_subset_cumsum  = np.nancumsum(x_subset) 
                total_prec = x_subset_cumsum[-1]
                sigma_time_array = np.zeros(shape = np.shape(x[:, slice_i_lat, slice_j_lon])[0]) #old was 48 instead of np.shape(x[:, slice_i_lat, slice_j_lon])[0]

                if total_prec == 0 : 
                    mean = 0 
                    sigma = 0
                    rcond = 0
                    pcond = 0
                    sigma_time = np.zeros(shape = np.shape(x[:, slice_i_lat, slice_j_lon])[0])
                    
                else : 
                    x_clean = x_subset_cumsum/total_prec
                    idx_rcond = np.where(x_clean > threshold)[0] # dynamically catch threshold from var_name.
                    rcond = np.nanmean(x_subset[idx_rcond])
                    sigma = len(x_subset[idx_rcond])/len(x_subset)
                    pcond = x_subset[idx_rcond[0]] # the precip value at which precips start to be acounted in Rcond
                    mean = np.nanmean(x_subset)
                    sigma_time = []
                    for t in range(np.shape(x[:, slice_i_lat, slice_j_lon])[0]):
                        x_t_subset_dirty = x[t, slice_i_lat, slice_j_lon].flatten()
                        x_t_subset = x_t_subset_dirty[~np.isnan(x_t_subset_dirty)]
                        sigma_t = np.nansum(x_t_subset>pcond)/len(x_t_subset)
                        sigma_time.append(sigma_t)
                    sigma_time_array = np.array(sigma_time)
                    
                sigma_global_time[i,j, :len(sigma_time_array)] = sigma_time_array
                mean_check[i,j] = mean
                sigma_global[i,j] = sigma
                rate_cond[i,j] = rcond
                precip_cond[i,j] = pcond

        output = [
                np.expand_dims(rate_cond, axis =-1), #here time must be last axis
                np.expand_dims(mean_check, axis =-1), 
                np.expand_dims(sigma_global, axis =-1), 
                np.expand_dims(precip_cond, axis = -1), 
                np.expand_dims(sigma_global_time, axis=-1)
                ]
    
        # expand so that we can concatenate futur dataarrays along day dim
        return output

    def make_labels_per_days_on_dict(self, dict):
        mcs = self.get_var_id_ds("MCS_label")
        mcs_labels_per_day = [np.unique(mcs.sel(days=day).MCS_label)[:-1] for day in mcs.days]

        # For each day get a list of all the labels that are vDCS
        labels_per_day = []
        missing_labels = []
        for labels in mcs_labels_per_day:
            vdcs_labels_today = []
            for label in labels : 
                if dict.get(label, False):
                    vdcs_labels_today.append(label)
                elif dict.get(label, "panini") == "panini":
                    missing_labels.append(label)
            labels_per_day.append(vdcs_labels_today)

        self.vdcs_labels_per_day = labels_per_day ## this should be generalized
        
        return labels_per_day, missing_labels
        
    def regrid_funcs_and_save_for_day(self, day, var_id='Prec', funcs=['max', 'mean']):
        """
        Compute multiple functions on new grid for a given day
        """	

        var_regridded_per_funcs = self.regrid_funcs_for_day(day, var_id=var_id, funcs_to_compute=funcs)
        var_ds = self.get_var_id_ds(var_id)
        da_day_per_keys = []
        for var_regridded in  var_regridded_per_funcs:
            if var_id in self.cloud_vars:  
                n_MCS = var_regridded.shape[3] # catch correct dimension here for labels_yxtm 
                da_day = xr.DataArray(var_regridded, dims=['lat_global', 'lon_global', 'days', 'MCS'], 
                                        coords={'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': [day], 'MCS':np.arange(n_MCS)})
            else :
                if isinstance(var_regridded, list):
                    da_day = xr.DataArray(var_regridded[0], dims=['lat_global', 'lon_global', 'days'], 
                                            coords={'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': [day]})
                else : 
                    da_day = xr.DataArray(var_regridded, dims=['lat_global', 'lon_global', 'days'], 
                                            coords={'lat_global': self.lat_global, 'lon_global': self.lon_global, 'days': [day]})
                    
            da_day_per_keys.append(da_day)

        del var_regridded_per_funcs
        gc.collect()
        
        for da_day, func in zip(da_day_per_keys, funcs):
            key = '%s_%s'%(func,var_id)
            var_ds[key].loc[{'days' : day}] = da_day
        
        file = self.get_var_ds_file(var_id)
        var_ds.to_netcdf(file)
        var_ds.close()
        
    def get_cond_prec_on_native_for_i_t(self, i_t, alpha_threshold = 85):
        file = self.get_var_ds_file("Prec")
        var = "threshold_cond_alpha_"+str(alpha_threshold)+"_Prec"
        with xr.open_dataset(file) as prec:
            threshold_prec = prec[var].values
            native_lon = self.slices_j_lon[-1][-1].stop
            native_lat = self.slices_i_lat[-1].stop
            out = np.zeros((native_lat, native_lon), float)
            i_day = self.casestudy.get_i_day_from_i_t(i_t)
            for i, slice_i_lat in enumerate(self.slices_i_lat):
                for j, slice_j_lon in enumerate(self.slices_j_lon[i]):
                    value_to_expand = threshold_prec[i,j, i_day]
                    out[slice_i_lat, slice_j_lon] = value_to_expand
        return out

    def safe_merge_and_save(self, ds_mcs, da_var, var_id, st_label_var_id, compat='override'):
        """
        Safely merges a DataArray into a Dataset and saves the merged Dataset to a NetCDF file.
        Includes checks for successful merge and file operations.

        Parameters:
        - ds_mcs: xarray.Dataset, the original dataset to merge into.
        - da_var: xarray.DataArray, the data array to merge.
        - var_id: str, the variable ID to rename `da_var` to in `ds_mcs`.
        - grid: object, should have a method `get_var_ds_file` returning the file path for `st_label_var_id`.
        - st_label_var_id: str, the label variable ID used to retrieve the file path from `grid`.
        - compat: str, compatibility mode for xarray.merge. Default is 'no_conflicts'.

        Returns:
        - bool: True if the operation was successful, False otherwise.
        """
        try:
            # Attempt to merge da_var into ds_mcs
            ds_mcs = xr.merge([ds_mcs, da_var.rename(var_id)], compat=compat)

            # Check if the merge was successful by comparing the merged dataset's variable with the original da_var
            if not ds_mcs[var_id].equals(da_var):
                logging.error("Merge failed: da_var values do not match in the merged dataset.")
                print("Try to replace instead")
                ds_mcs[var_id] = da_var.rename(var_id)

            # Retrieve the file path
            file_mcs_ds = self.get_var_ds_file(st_label_var_id)

            # Check if the file exists before attempting to delete
            if os.path.exists(file_mcs_ds):
                os.remove(file_mcs_ds)
                # print(os.path.exists(file_mcs_ds))
            else:
                logging.warning(f"File {file_mcs_ds} does not exist, nothing to delete.")

            print(ds_mcs[var_id].equals(da_var))
            # Save the merged dataset to the file
            ds_mcs.to_netcdf(file_mcs_ds)

        except Exception as e:
            logging.exception(f"An error occurred during the merge and save process: {e}")
            return False

        return True
    
    def build_cloud_intersect(self, cloud_type, coverage_threshold=0.5):
        """
        Makes CS overlap bool map with  cond_Prec_85
        """
        # cloud_types = ["clouds_cond_prec_15", "vdcs_cond_prec_15", "mcs_cond_prec_15"]
        # assert cloud_type in cloud_types
        cloud_intersect_var_id = "intersection_"+cloud_type
        
        sigma_85 = self.get_var_id_ds("Prec").Sigma_cond_alpha_85_Prec
        cs_cond = self.get_var_id_ds(cloud_type)
        sigma_cs = cs_cond.Rel_surface.sum(axis = 3)
        cs_cond[cloud_intersect_var_id] = sigma_cs/sigma_85
        bool_covered = cs_cond[cloud_intersect_var_id]>coverage_threshold
        cs_cond["intersection_over_half_"+cloud_intersect_var_id] = bool_covered ## add this 
        mcs_cond_file = self.get_var_ds_file(cloud_type)
        os.remove(mcs_cond_file)
        cs_cond.to_netcdf(mcs_cond_file)
        
    def make_region_mask(self, min_lat, max_lat, min_lon, max_lon, pre_mask = True):
        """
        # warmpool = make_region_mask(0, 25, 125, 185, pre_mask = self.mask_ocean)
        # indian_ocean = make_region_mask(-15, 10, 50, 100, pre_mask = self.mask_ocean)
        # rest_ocean = np.logical_and(~warmpool, self.mask_ocean)
        """
        array_shape = (60, 360, 30) # 40 days in this sim
        # Combine the masks along each dimension
        mask = np.zeros(array_shape, dtype=bool)
        mask[min_lat+30:max_lat+30, min_lon:max_lon, :] = True
        mask = np.logical_and(mask, pre_mask)
        
        return mask


