import numpy as np
import os
import time
import sys
import glob
import pickle
import time
import math 
import xarray as xr 
import pandas as pd
import gc 

import warnings
# from .toocan_loaders.load_toocan import load_toocan
from .toocan_loaders.load_toocan_sam import load_toocan_sam
from .toocan_loaders.load_toocan_mcsmip import load_TOOCAN

import warnings
from scipy.optimize import OptimizeWarning

from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

class StormTracker():
    """
    To instantiate Toocan's object only once per jd.
    """
    def __init__(self, casestudy, label_var_id = "MCS_label", overwrite_storms = False, overwrite=False, verbose=False):
        
        self.settings = casestudy.settings
        
        self.region = self.settings['REGION']
        self.model = self.settings['MODEL']
        self.name = f"{self.model}_{self.region}"

        start_time = time.time()
        self.overwrite = overwrite 
        self.verbose = verbose
        self.label_var_id = label_var_id

        if self.label_var_id in ["MCS_label", "MCS_Feng", "Conv_MCS_label", "vDCS"]:
            self.dir_storm = self.settings['DIR_STORM_TRACKING']
        elif self.label_var_id == 'MCS_label_Tb_Feng':
            self.dir_storm = self.settings['DIR_STORM_TRACKING_TB_FENG']

        self.Utime_step = self.settings["NATIVE_TIMESTEP"] 
        self.i_t_start = self.settings["TIME_RANGE"][0] + 816673 # it seems that storms UTC start from this very high number so I'm adapting it here
        self.i_t_end = self.settings["TIME_RANGE"][1] + 816673

        # get storm tracking data
        if self.verbose : print("Loading storms...")
        if True :
            self.ds_storms, self.file_storms = self.load_storms_tracking(overwrite_storms)
        else : 
            self.ds_storms = self.load_storms_tracking_from_netcdf()
                
        if self.verbose : print(f"Time elapsed for loading storms: {time.time() - start_time:.2f} seconds")
    
        ## test and incubate in a func
        if self.overwrite:
            grw_var_names= ["growth_rate", "r_squared", "growth_r_squared", "decay_r_squared", "t0", "t_max", "t_f", "s_max"]
            grw_vars = [ [] for _ in grw_var_names]
            with xr.open_dataset(self.file_storms) as ds_storms:
                for label in ds_storms.DCS_number:
                    storm = ds_storms.sel(DCS_number = label)
                    grw_output = self.get_storm_growth_rate(storm, r_treshold=0.85, verbose = True)
                    for out, grw_var in zip(grw_output, grw_vars):
                        grw_var.append(out)
                grw_arr_vars = [np.array(grw_var) for grw_var in grw_vars]
                grw_data_vars = [xr.DataArray(grw_arr_var, 
                                            dims = ['DCS_number'], 
                                            coords = {"DCS_number": ds_storms.DCS_number})            
                                                for grw_arr_var in grw_arr_vars]
                for grw_var_name, grw_data_var in zip(grw_var_names, grw_data_vars):
                    ds_storms[grw_var_name] = grw_data_var
                self.save_storms(ds_storms)

        self.valid_3d = casestudy.get_valid_3d()

    def load_storms_tracking(self, overwrite):
        ## Make it a netcdf so that we don't struggle with loading it anymore
        dir_out = os.path.join(self.settings["DIR_DATA_OUT"], self.name)
        file_storms = os.path.join(dir_out, "storms_"+self.label_var_id+".nc")
        model_name = self.settings["MODEL"].split("_")[0]

        if os.path.exists(file_storms) and not overwrite:
            if self.verbose : print("loading storms from netcdf")
            with open(file_storms, 'rb') as file:
                ds_storms = xr.open_dataset(file)
        else : 
            if overwrite : 
                if self.verbose : print("Loading storms again because overwrite_storms is True")
            paths = glob.glob(os.path.join(self.dir_storm, '*.gz'))
            # For dyamond 2 there are many filetracking for other models in the dir_storm
            toocan_paths = []
            for path in paths : 
                path_model = path.split("-")[1]
                if path_model == model_name:
                    toocan_paths.append(path)
                    
                elif self.settings["MODEL"] == "SAM3d": ## this is only for sam highres now
                    if "SAM" in path :
                        toocan_paths.append(path)

            ## not the same loader here 
            if self.settings["MODEL"] == "DYAMOND_SAM_post_20_days" or self.settings["MODEL"] == "SAM_4km_30min_30d" or self.settings["MODEL"] == "SAM3d":
                storms = load_toocan_sam(toocan_paths[0])+load_toocan_sam(toocan_paths[1])
            elif "lowRes" in self.settings["MODEL"]:
                storms = load_TOOCAN(toocan_paths[0])+load_TOOCAN(toocan_paths[1])

            # Extract the relevant variables

            ## differentiating clouds from vdcs here 
            filtered_storms = []
            lat_min, lat_max = self.settings["BOX"][0], self.settings["BOX"][1]
            for storm in storms:
                if storm.INT_UTC_timeEnd / self.Utime_step >= self.i_t_start:
                    ## chat codes heres 
                    lat_in_range = any(lat_min-2 <= lat <= lat_max+2 for lat in storm.clusters.LC_lat)
                    if lat_in_range:
                    # Add the storm to the new list if the condition is met
                        if "lowRes" in self.settings["MODEL"] and self.label_var_id == "MCS_Feng":
                            if storm.INT_classif_MCS : filtered_storms.append(storm)
                        elif self.label_var_id == "vDCS":
                            if  (storm.INT_TbMin<210) & (storm.INT_duration > 2.5) & (storm.INT_velocityAvg < 20):
                                filtered_storms.append(storm)
                        else : 
                            filtered_storms.append(storm)

            # Update the original list with the filtered storms
            storms = filtered_storms
            print("making ds storms ...")
            ## !!!!!!!!!!!!
            print()
            ds_storms = self.make_ds(storms) ## this drop the duplicates
            print("making duplicates ds storms :")
            clean_storms = self.clean_duplicate_storms(storms)
            if len(clean_storms)>0:
                ds_storms_duplicate = self.make_ds(clean_storms)
                print("Merging storms and cleaned duplicates")
                ds_storms = xr.concat([ds_storms, ds_storms_duplicate], dim ="DCS_number")
            else : 
                print("All duplicates removed or no duplicates")
            print("Saving storms")
            ds_storms.to_netcdf(file_storms)
            del filtered_storms
            gc.collect()
            print("ds storms saved !")

        # label_storms = [ds_storms['label'].isel(i) for i in range(len(ds_storms['label']))]
        # dict_i_storms_by_label = {}
        # for i, label in enumerate(ds_storms['label']):
        #     if label not in dict_i_storms_by_label.keys():
        #         dict_i_storms_by_label[label] = i

        return ds_storms, file_storms #, label_storms, dict_i_storms_by_label

    def clean_duplicate_storms(self, storms):
        l, d = [], []
        for storm in storms:
            l.append(storm.DCS_number)
            d.append(storm.INT_localtime_Init)

        def get_duplicates(l):
            seen = set()
            duplicates = set()
            for item in l:
                if item in seen:
                    duplicates.add(item)
                else:
                    seen.add(item)
            return list(duplicates)

        ll= get_duplicates(l)
        print(f"Found {len(ll)} duplicates storms.")
        ss = [storm for storm in storms if storm.DCS_number in ll]

        ssll = []
        for i, l in enumerate(ll):
            ssll.append([])
            k=0
            while k <2:
                for s in ss:
                    if s.DCS_number == l:
                        ssll[i].append(s)
                        k+=1
        bool_ss= []
        for i, ss in enumerate(ssll):
            bool_ss.append(ss[0] == ss[1]) #need to add __eq__ to toocan_loader classes for this to work
        
        if len(bool_ss)>0 : 
            print(f"Found {len(ssll)} duplicates, removed {np.round(1 - (sum(bool_ss)/len(bool_ss)), 2)} of them")
            storms_clean = []
            for i, storm_bool in enumerate(bool_ss):
                if storm_bool : 
                    storms_clean.append(ssll[i][0])

            return storms_clean
        else : 
            return []

    def save_storms(self, ds_storms):
        os.remove(self.file_storms)
        ds_storms.to_netcdf(self.file_storms)
            
    def _piecewise_linear(self, t:np.array,t_breaks:list,s_max:float):
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

    def _piecewise_fit(self, t:np.array,s:np.array,t_breaks_0:list,s_max_0:float):    
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
            return self._piecewise_linear(t,p[0:N_breaks],p[-1])

        # we add bounds so that time breaks stay ordered
        t_lower_bounds = [0] + t_breaks_0[:-1] # 0 instead of -np.inf
        t_upper_bounds = t_breaks_0[1:] + [np.inf]
        
        s_lower_bounds = [0] # Positive or null surfaces. 
        s_upper_bounds = [+np.inf] #null surfaces at first and last breaks
        
        p , e = curve_fit(piecewise_fun, t, s, p0=t_breaks_0+[s_max_0], bounds = (t_lower_bounds + s_lower_bounds, t_upper_bounds + s_upper_bounds))

        s_id = self._piecewise_linear(t, p[0:N_breaks], p[-1])
        s_max = p[-1]
        t_breaks = list(p[0:N_breaks])
        
        return t_breaks,s_max,s_id

    def get_storm_growth_rate(self, storm, r_treshold = 0.85, verbose = False, plot = False):
        """
        Given a storm object, update it's growth_rate attribute 
        Returns an ax object to plot the fit
        """
        u_i, u_e = storm.INT_UTC_timeInit.values, storm.INT_UTC_timeEnd.values
        init= max(int(u_i/self.Utime_step) - self.i_t_start , 0)
        end = min(int(u_e/self.Utime_step) - self.i_t_start , self.i_t_end - 1)+1
        surf = storm.LC_surfkm2_241K.values[init:end]
        nan_output = [np.nan for _ in range(8)]

        if len(surf) <= 4 : 
            # setattr(storm, 'growth_rate', growth_rate)
            if verbose : print("A very short-lived storm passed by here...")
            return nan_output
        else : 
            time = np.arange(0, len(surf))

            s_max = max(surf)
            time_breaks = [0, len(surf)//2, len(surf)]

            warnings.filterwarnings("error", category=OptimizeWarning)

            try: # parfois ça marche pô
                t_breaks, s_max, s_id = self._piecewise_fit(time, surf, time_breaks, s_max)
                t0, t_max, t_f = t_breaks
                r_squared = r2_score(surf, s_id)
                growth_r_squared = r2_score(surf[:math.ceil(t_breaks[1])], s_id[:math.ceil(t_breaks[1])])
                decay_r_squared = r2_score(surf[math.floor(t_breaks[1]):], s_id[math.floor(t_breaks[1]):])
                growth_rate = s_max / (t_breaks[1] - t_breaks[0])

            except OptimizeWarning as e:
                print("That's a complicated storm")
                return nan_output
            
            except Exception as e:
                # Handle the warning here, e.g., print a message or log it
                print("Caught Exception:", e)
                return nan_output
            
            if verbose : print(f"For storm with label {storm.DCS_number}, the growth rate computed by fitting a triangle is {growth_rate} with an r-score of {r_squared}")

            return [growth_rate, r_squared, growth_r_squared, decay_r_squared, t0, t_max, t_f, s_max]
        
    def make_ds(self, storms):
        """
        TODO : the whole clusters part seems to only return np.nan... so not working
        TODO : Replace 1800 and 960 by Utime_step and i_t_start, model dependent
        """

        attribute_names = [attr for attr in dir(storms[0]) if not callable(getattr(storms[0], attr)) and  not attr.startswith("clusters") and not attr.startswith("__")]
        print("Label wise attributes ", attribute_names)
        dict_storm = {attr: [getattr(obj, attr) for obj in storms if not attr.startswith("__")] for attr in attribute_names}

        ## get clusters per storm so per label
        clusters = [storm.clusters for storm in storms]

        #first one for attr names
        mcs_lfc = clusters[0]
        clusters_attribute_names = [attr for attr in dir(mcs_lfc)if not callable(getattr(mcs_lfc, attr)) and not attr.startswith("__")]
        print("Label's Lifecycle wise attributes ", clusters_attribute_names)

        ## Build ds for storms attributes
        data_vars = {}
        labels = dict_storm["DCS_number"]
        for key in dict_storm.keys():
            da = xr.DataArray(dict_storm[key], dims = ['DCS_number'], coords = {'DCS_number' : labels})
            data_vars[key] = da
            
        # ds_storms = xr.Dataset(data_vars)

        ## Build ds for clusters attributes (list)
        storm_clusters = {key : [] for key in clusters_attribute_names}
        for i_storm, mcs_lfc in enumerate(clusters):
            for attr in clusters_attribute_names:
                var = getattr(mcs_lfc, attr)
                if len(var)>0:
                    storm_clusters[attr].append(var)
                    
        Utime_min, Utime_max = 0,0
        for storm in storms : 
            Utime_min = min(Utime_min, storm.INT_UTC_timeInit)
            Utime_max = max(Utime_max, storm.INT_UTC_timeEnd)  
            
        Utime = np.arange(Utime_min, Utime_max + self.Utime_step, self.Utime_step)  
        time = np.arange(self.i_t_start, self.i_t_end, 1) #should be similar

        n_label = len(storm_clusters["LC_UTC_time"])
        n_time = len(time) # all utime values possible within timerange

        for key in storm_clusters.keys():
            # one cluster attributes is empty
            if len(storm_clusters[key])>0 : 
                data = np.full((n_label, n_time), np.nan)
                for i_label, value_list, index_Utime in zip(np.arange(len(storm_clusters[key])), storm_clusters[key], storm_clusters["LC_UTC_time"]):
                    for i, utime in enumerate(index_Utime): # peut-être relaxé pour utilise INT_UTC_timeInit/1800 : INT_UTC_timeEnd/1800
                        idx_utime = int(utime/self.Utime_step)-self.i_t_start # Utime
                        if idx_utime >= 0 and idx_utime < self.i_t_end - self.i_t_start: 
                            data[i_label, idx_utime] = value_list[i] 
                            # if idx_utime == 0  : print(i_label)
                da = xr.DataArray(data, dims = ['DCS_number', 'time'], coords = {'DCS_number' : dict_storm['DCS_number'], 'time' : time})
                    
                # handle same name 
                if key =="olrmin":
                    data_vars["lifecycle_olrmin"] = da
                else : 
                    data_vars[key] = da

        ds = xr.Dataset(data_vars)
        ## en théorie plus besoin, si de nouveaux doublons n'aparaissent pas
        # Sont apparus, 16k dans sam_summer
        ds = ds.drop_duplicates('DCS_number', keep = False) #choix ici, attendre rep Laurent

        return ds
        
    def get_vdcs_dict(self):
        storms = xr.open_dataset(self.file_storms)
        ## Filter for vDCS on storms object, must be MCS_label to capture them all
        vDCS = (storms["INT_TbMin"]<210) & (storms["INT_duration"] > 2.5) & (storms["INT_velocityAvg"] < 20)

        ## make a dict that says if yes or no this label is a vDCS
        storms["BOOL_VDC"] = vDCS
        bool_vdc = storms.BOOL_VDC.to_pandas().to_list()
        labels = storms.DCS_number.to_pandas().to_list()
        bool_dict_label = dict(zip(labels, bool_vdc))
        storms.close()
        return bool_dict_label

    def get_frame_data(self, ft, iDCS):
        """
        returns everything to compute the frame and fields of a DCS over it's lifetime
        start and end corresponds to filetracking time index
        time_array corresponds to i_t of global variables (seg_mask, precip, U,...)
        """

        storm = ft.sel(DCS_number = iDCS)
        start = np.max([0,int(storm.INT_UTC_timeInit.values/self.settings["NATIVE_TIMESTEP"])-self.i_t_start])
        end = np.min(1+int(storm.INT_UTC_timeEnd.values/self.settings["NATIVE_TIMESTEP"])-self.i_t_start, self.i_t_end-self.i_t_start)
        lon_array, lat_array = storm.LC_lon[start:end].values, storm.LC_lat[start:end].values
        velocity_array, time_array = storm.LC_velocity[start:end].values, storm.LC_UTC_time[start:end].values/1800 - self.i_t_start

        # Initialize speed arrays with zeros
        speed_lon = np.zeros_like(lon_array)
        speed_lat = np.zeros_like(lat_array)
        for i in range(1, len(lon_array) - 1):
            speed_lon[i] = (lon_array[i + 1] - lon_array[i - 1]) / 2 #*delta_t, for now units are not m/s (which is a shame)
            speed_lat[i] = (lat_array[i + 1] - lat_array[i - 1]) / 2 #*delta_t


        assert time_array[0]==start
        if time_array[-1] +1 != end:
            print("is this an end MCS ? ", iDCS)

        time_array = time_array +self.settings["TIME_RANGE"][0]

        time_smax = -1
        if np.any(storm["LC_surfkm2_241K"] == storm["INT_surfmaxkm2_241K"]):
            time_smax = storm.LC_UTC_time[storm["LC_surfkm2_241K"] == storm["INT_surfmaxkm2_241K"]][0]
            time_smax = int(time_smax.values/1800 - self.i_t_start)
        
        i_smax = -1
        i_smax_argwhere = np.argwhere(time_array==(time_smax + self.settings["TIME_RANGE"][0]))
        if i_smax_argwhere.size >0       : 
            i_smax = i_smax_argwhere[0][0]

        mask = np.isin(time_array, list(self.valid_3d))
        ## Obviouvsly these outputs must be simplified
        return start, end, lon_array, lat_array, velocity_array, time_array, time_smax, i_smax, lon_array[mask], lat_array[mask], velocity_array[mask], time_array[mask], speed_lon[mask], speed_lat[mask]

    def get_extents_slices(self, lons, lats, large_scale_frame_size=4):
        lons_min, lons_max = lons-large_scale_frame_size, lons+large_scale_frame_size
        lats_min, lats_max = lats-large_scale_frame_size, lats+large_scale_frame_size
        extents = [[lon_min, lon_max, lat_min, lat_max] for lon_min, lon_max, lat_min, lat_max in zip(lons_min, lons_max, lats_min, lats_max)]
        slices_lon = [slice(lon_min, lon_max) for lon_min, lon_max in zip(lons_min, lons_max)]
        slices_lat = [slice(lat_min, lat_max) for lat_min, lat_max in zip(lats_min, lats_max)]
        return extents, slices_lon, slices_lat




    # if plot : 
    #     # Return ax object if plotting is necessary
    #     fig, ax = plt.subplots()
    #     ax.scatter(time, surf, label='Surface')
    #     time_plot = np.linspace(0, time.max(), 1000)
    #     #ax.plot(time_plot, piecewise_linear(time_plot, t_breaks, s_breaks), 'r-', label='Idealized Surface')
    #     ax.legend()
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('Surface Values')
    #     ax.set_title('Fitting a Triangle Function to Surface Values over Time')
    #     plt.show()

    # def load_storms_tracking_from_netcdf(self):
    #     # dir_out = os.path.join(self.settings["DIR_DATA_OUT"], self.grid.casestudy.name)
    #     dir_storms_netcdf = self.settings["DIR_STORM_TRACKING_NETCDF"]
    #     paths = glob.glob(os.path.join(dir_storms_netcdf, '*.nc'))
    #     model_name = self.settings["MODEL"].split("_")[0]
    #     if model_name == "OBS" : model_name = "OBSv7" #bc V6 still pollutes
    #     dataset_paths = []
    #     for path in paths : 
    #         path_model = path.split("-")[1]
    #         if path_model == model_name:
    #             dataset_paths.append(path)
    #     ds_storms = xr.open_mfdataset(dataset_paths, combine = 'nested')
        
        # ds1, ds2 = xr.align(datasets[0], datasets[1], join = 'outer') #supposedly only two in this experiment, adapth with zip and ** otherwise
        # ds1_stacked = ds1.stack(time_DCS = ("time", "DCS"))
        # ds2_stacked = ds2.stack(time_DCS= ("time", "DCS"))
        # concat_ds = xr.concat([ds1_stacked, ds2_stacked], dim="time_DCS")
        # ds_storms = concat_ds.unstacked('time_DCS')
        # ds_storms = xr.concat(datasets, dim = 'time')
        
        # return ds_storms  
    