import pandas as pd
import os 
import sys
import re
import yaml 
import numpy as np
import datetime as dt
import gc
import xarray as xr
import warnings 
import time
import subprocess

from work.thermo import saturation_specific_humidity
from work import storm_tracker

class Handler():
    def __init__(self, settings_path):
        ## could add whole casestudy here but whatever
        self.settings_path = settings_path
        with open(self.settings_path, 'r') as file:
            self.settings = yaml.safe_load(file)
        self.dict_date_ref = self.settings["DATE_REF"]
        # self.rel_table = self.load_rel_table(self.settings['REL_TABLE'])

    def shift_lon(self, ds):
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
        ds = ds.sortby('lon')
        return ds
    
    def shift_longitude(self, ds):
        ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
        ds = ds.sortby('longitude')
        return ds

    def adjust_longitude(self, lon):
        if lon < 0:
            return lon + 360
        else:
            return lon
        
    def run_ncks(self, ncks_command, temp_file):
        subprocess.run(ncks_command, shell=True)
        var = xr.open_dataset(temp_file)
        return var

    def handle_longitude_slicing(self, lon_min_index, lon_max_index, ncks_base_command, temp_file, filepath_var, temp_var, data_lon_values):
        if lon_min_index <= lon_max_index:
            # Longitude range does not cross the 360-degree line
            str_lon_slice = f"{lon_min_index},{lon_max_index},1"  # Include stride for index-based slicing
            ncks_command = f"{ncks_base_command} -d lon,{str_lon_slice} {filepath_var} {temp_file}"
            subprocess.run(ncks_command, shell=True)
            var = xr.open_dataset(temp_file)
            return var
        else:
            # Longitude range crosses the 360-degree line
            # First part: lon_min_index to end of array
            temp_file1 = os.path.join(temp_var, "temp_part1.nc")
            lon_max_index1 = len(data_lon_values) - 1 
            str_lon_slice1 = f"{lon_min_index},{lon_max_index1},1"
            ncks_command1 = f"{ncks_base_command} -d lon,{str_lon_slice1} {filepath_var} {temp_file1}"
            subprocess.run(ncks_command1, shell=True)

            # Second part: start of array to lon_max_index
            temp_file2 = os.path.join(temp_var, "temp_part2.nc")
            lon_min_index2 = 0
            str_lon_slice2 = f"{lon_min_index2},{lon_max_index},1"
            ncks_command2 = f"{ncks_base_command} -d lon,{str_lon_slice2} {filepath_var} {temp_file2}"
            subprocess.run(ncks_command2, shell=True)
            # merge
            ds1 = xr.open_dataset(temp_file1)
            ds2 = xr.open_dataset(temp_file2)
            var = xr.concat([ds1, ds2], dim='lon')
            return var

    def i_t_to_nice_datetime(self, i_t):
        dict_date_ref = self.settings["DATE_REF"]
        datetime_ref = dt.datetime(dict_date_ref['year'], dict_date_ref['month'], dict_date_ref['day'])
        timestamp_ref = datetime_ref.timestamp()

        i_t_in_seconds = i_t * 30 * 60
        timezone_weird_lag_to_watch = 2*60*60 #2hours
        timestamp = timestamp_ref + i_t_in_seconds + timezone_weird_lag_to_watch
        date = dt.datetime.utcfromtimestamp(timestamp)
        
        string_date = date.strftime("%Y_%m_%d")
        year, month, day = string_date.split("_")
        hours = int(date.strftime("%H"))
        minutes = int(date.strftime("%M"))
        n_half_hours = int(2*hours + minutes/30 + 1)
        
        datetime_ref = dt.datetime(int(year), int(month), int(day), hours, minutes)
        formatted_date = datetime_ref.strftime("%d %B %Y, %H:%M")

        return formatted_date


    def load_var(self, casestudy, var_id, i_t, z_idx = None, sel_dict = None): 
        """
        Load a variable at specified i_t.
        If the variable is a new one, calls the appropriate function, that will recursively call load_var
        If the variable is 3D one, it will return a dataset instead.
            Must be handled in your designed funcs that depends on 3D vars

        I don't want it to load the data anymore as i will ue a subsequent slicing dict to avoid loading
        """
        new_var_names = casestudy.new_variables_names
        var_2d = casestudy.var_names_2d
        var_3d = casestudy.var_names_3d
        new_var_funcs = casestudy.new_var_functions

        if var_id in new_var_names:
            if hasattr(self, new_var_funcs[var_id]):
                load_func = getattr(self, new_var_funcs[var_id])
            else : print(f"Handler has no method {new_var_funcs[var_id]}")
            da_new_var = load_func(casestudy, i_t, sel_dict) 
            da_new_var = self.shift_lon(da_new_var)
            return da_new_var
            
        else : 
            path_data_in = casestudy.settings["DIR_DATA_2D_IN"]
            if self.settings["MODEL"] in ["DYAMOND_SAM", "SAM_4km_30min_30d", "SAM3d"]:
                root = self.get_rootname_from_i_t(i_t)
                filename_var = root+f".{var_id}.2D.nc"
                filepath_var = os.path.join(path_data_in, filename_var)
                if var_id in var_2d:
                                var = xr.open_dataarray(filepath_var)
                                var = self.shift_lon(var)  # Shift longitudes before selection
                                var = var.sel(sel_dict)

                elif var_id in var_3d :
                    path_data_in = casestudy.settings["DIR_DATA_3D_IN"]
                    # chunks = {'z': 74} # Always 74 vertical level in these data
                    filename_var = root+f"_{var_id}.nc"
                    filepath_var = os.path.join(path_data_in, filename_var)
                    temp = os.path.join(self.settings["DIR_TEMPDATA"], casestudy.name) # ?
                    temp_var = os.path.join(temp, var_id)
                    if not os.path.exists(temp_var):
                        os.makedirs(temp_var)

                    adj_sel_lon_start = self.adjust_longitude(sel_dict['lon'].start) ## I might need to get that out of this elif
                    adj_sel_lon_stop = self.adjust_longitude(sel_dict['lon'].stop)

                    test = xr.open_dataset("/bdd/DYAMOND/SAM-4km/OUT_3D/DYAMOND_9216x4608x74_7.5s_4km_4608_0000001440_PP.nc")
                    data_lon_values = test.lon.values.copy()
                    data_lat_values = test.lat.values.copy()
                    z_levels = test.z.values.copy()
                    test.close()

                    # Find latitude indices (assuming latitude does not wrap around)
                    lat_indices = np.where((data_lat_values >= sel_dict['lat'].start) & (data_lat_values <= sel_dict['lat'].stop))[0]
                    lat_min_index, lat_max_index = lat_indices[[0, -1]]

                    lat_min, lat_max = np.where((test.lat.values > sel_dict['lat'].start) & (test.lat.values < sel_dict['lat'].stop))[0][[0, -1]]

                    # str_lon_slice = f"{lon_min},{lon_max}" ## this is now sorted out by handle_longitude_slicing
                    str_lat_slice = f"{lat_min},{lat_max}"
                    
                    ncks_base_command = f"ncks -O -d lat,{str_lat_slice} -d time,0"

                    # Handle longitude indices
                    if adj_sel_lon_start <= adj_sel_lon_stop:
                        # Longitude range does not cross 360 degrees
                        lon_indices = np.where((data_lon_values >= adj_sel_lon_start) & (data_lon_values <= adj_sel_lon_stop))[0]
                        lon_min_index, lon_max_index = lon_indices[[0, -1]]
                    else:
                        # Longitude range crosses 360 degrees
                        lon_indices_part1 = np.where((data_lon_values >= adj_sel_lon_start) & (data_lon_values <= 360))[0]
                        lon_indices_part2 = np.where((data_lon_values >= 0) & (data_lon_values <= adj_sel_lon_stop))[0]
                        lon_indices = np.concatenate((lon_indices_part1, lon_indices_part2))
                        lon_min_index = lon_indices[0]
                        lon_max_index = lon_indices[-1]

                    str_lat_slice = f"{lat_min_index},{lat_max_index},1"

                    ncks_base_command = f"ncks -O -d lat,{str_lat_slice} -d time,0"

                    if isinstance(z_idx, int):
                        ncks_base_command += f" -d z,{z_idx}"
                        temp_file = os.path.join(temp_var, f"z_ind_{z_idx}.nc")
                        var = self.handle_longitude_slicing(lon_min_index, lon_max_index, ncks_base_command, temp_file, filepath_var, temp_var, data_lon_values)
                    elif z_idx == "all":
                        temp_file = os.path.join(temp_var, "z_all.nc")
                        var = self.handle_longitude_slicing(lon_min_index, lon_max_index, ncks_base_command, temp_file, filepath_var, temp_var, data_lon_values)
                    elif z_idx == "L'altitude de la troposphère":
                        z_min, z_max = 0, len(z_levels) - 23  # Adjust based on your z_levels
                        ncks_base_command += f" -d z,{z_min},{z_max},1"
                        temp_file = os.path.join(temp_var, "z_tropo.nc")
                        var = self.handle_longitude_slicing(lon_min_index, lon_max_index, ncks_base_command, temp_file, filepath_var, temp_var, data_lon_values)
                    else : 
                        print("We didn't understood what was your z_idx")
            else : 
                print("You failed to get the var, you'll get a bug !")

            var = self.shift_lon(var)
            return var
         
    def load_seg(self, i_t, sel_dict):
        path_toocan = self.get_filename_classic(i_t) ## There is the differences
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=xr.SerializationWarning)
            img_toocan = xr.open_dataset(path_toocan, engine='netcdf4').DCS_number #.sel(latitude = slice(self.settings["BOX"][0], self.settings["BOX"][1]))# because otherwise goes to -40, 40
        img_toocan = self.shift_longitude(img_toocan).sel(sel_dict)
        return img_toocan

    def load_conv_seg(self, grid, i_t):
        path_toocan = self.get_filename_classic(i_t) ## There is the differences

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=xr.SerializationWarning)
            img_toocan = xr.open_dataset(path_toocan, engine='netcdf4').cloud_mask.sel(latitude = slice(self.settings["BOX"][0], self.settings["BOX"][1]))# because otherwise goes to -40, 40

        img_labels = np.unique(img_toocan)[:-1]

        # reload storm everytime, fuck it.. dependencies might be doomed
        st = storm_tracker.StormTracker(grid, label_var_id = "MCS_label", overwrite = False) # takes 2sec with all overwrite to false

        ds_storm = xr.open_dataset(st.file_storms)

        valid_labels = ds_storm.label

        img_valid_labels = [label for label in img_labels if label in valid_labels]

        print(len(img_valid_labels))
        for label in img_valid_labels :
            # get storm dataarrays
            storm = ds_storm.sel(label = label)
            if storm.r_squared.values >= 0.8:
                # retrieve time_init in ditvi format
                time_init = storm.Utime_Init.values/self.settings["NATIVE_TIMESTEP"]
                # compute growth init from growth_rate t0 fit
                growth_init = np.round(time_init + storm.t0.values, 2)
                # compute growth end from growth_rate t_max fit
                growth_end = np.round(time_init + storm.t_max.values, 2)
                # End of MCS life time in ditvi format for check
                time_end = storm.Utime_End.values/self.settings["NATIVE_TIMESTEP"]
                # print(label, growth_init, i_t, growth_end, time_end)
                if i_t >= growth_init and i_t <= growth_end:
                    pass
                else : 
                    img_toocan = img_toocan.where(img_toocan != label, np.nan)
            else : 
                img_toocan = img_toocan.where(img_toocan != label, np.nan)
        img_toocan = self.shift_longitude(img_toocan)
        return img_toocan

    def load_filter_vdcs_seg(self, grid, i_t):
        img_toocan = self.load_seg(grid, i_t)
        img_labels = np.unique(img_toocan)[:-1] if np.any(np.isnan(img_toocan)) else np.unique(img_toocan)
        # reload storm everytime, fuck it.. dependencies might be doomed
        st = storm_tracker.StormTracker(grid, label_var_id = "MCS_label", overwrite_storms = False, overwrite = False) # takes 2sec with all overwrite to false
        dict = st.get_vdcs_dict()
        valid_labels_per_day, _ = grid.make_labels_per_days_on_dict(dict)
        for i_day, day in enumerate(grid.casestudy.days_i_t_per_var_id[st.label_var_id].keys()):
            if i_t in grid.casestudy.days_i_t_per_var_id[st.label_var_id][day]:
                current_day = day
                current_i_day = i_day
                break
        today_valid_labels = valid_labels_per_day[i_day]
        
        for current_label in img_labels: 
            if current_label not in today_valid_labels:
                img_toocan = img_toocan.where(img_toocan != current_label, np.nan)

        img_toocan = self.shift_longitude(img_toocan)

        return img_toocan

    def load_seg_tb_feng(self, grid, i_t):
        path_toocan = self.get_filename_tb_feng(i_t) ## There is the differences
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=xr.SerializationWarning)
            img_toocan = xr.open_dataset(path_toocan, engine='netcdf4').cloud_mask.sel(latitude = slice(self.settings["BOX"][0], self.settings["BOX"][1]))

        img_toocan = self.shift_longitude(img_toocan)

        return img_toocan


    def get_timestamp_from_filename(self, filename):
        """
        input: the filename of the classical DYAMOND .nc data file "DYAMOND_9216x4608x74_7.5s_4km_4608_0000345840.U10m.2D.nc"
        output: 345840
        """
        timestamp_pattern = r'_(\d{10})\.\w+\.2D\.nc'
        match = re.search(timestamp_pattern, filename)
        if match : 
            timestamp = match.group(1)
            return timestamp
        else : return None

    def get_rootname_from_i_t(self, i_t):
        """
        input: the i_t of the classical DYAMOND .nc data file, eg. 1441 (*240 = 345840)
        output: data rootname eg. "DYAMOND_9216x4608x74_7.5s_4km_4608_0000345840"
        """

        string_timestamp = str(int(int(i_t) * 240)).zfill(10) # 240 ? 
        result = f"DYAMOND_9216x4608x74_7.5s_4km_4608_"+string_timestamp
        return result

    def get_filename_classic(self, i_t):
        root = self.settings['DIR_STORM_TRACK']

        dict_date_ref = self.settings["DATE_REF"]
        datetime_ref = dt.datetime(dict_date_ref['year'], dict_date_ref['month'], dict_date_ref['day'])
        timestamp_ref = datetime_ref.timestamp()

        i_t_in_seconds = i_t * 30 * 60
        timezone_weird_lag_to_watch = 2*60*60 #2hours
        timestamp = timestamp_ref + i_t_in_seconds + timezone_weird_lag_to_watch
        date = dt.datetime.utcfromtimestamp(timestamp)
        
        string_date = date.strftime("%Y_%m_%d")
        hours = int(date.strftime("%H"))
        minutes = int(date.strftime("%M"))
        n_half_hours = int(2*hours + minutes/30 + 1)
        dir_path = os.path.join(root, string_date)
        string_date_no_underscore = string_date.replace('_', '')
        file_root= "ToocanCloudMask_SAM_"+string_date_no_underscore+'-'+str(n_half_hours).zfill(3)+'.nc'
        filename = os.path.join(dir_path, file_root)
        return filename

    def get_filename_tb_feng(self, i_t):
        root = self.settings['DIR_STORM_TRACK_TB_FENG']

        dict_date_ref = self.settings["DATE_REF"]
        datetime_ref = dt.datetime(dict_date_ref['year'], dict_date_ref['month'], dict_date_ref['day'])
        timestamp_ref = datetime_ref.timestamp()

        i_t_in_seconds = i_t * 30 * 60
        timezone_weird_lag_to_watch = 2*60*60 #2hours
        timestamp = timestamp_ref + i_t_in_seconds + timezone_weird_lag_to_watch
        date = dt.datetime.utcfromtimestamp(timestamp)
        
        string_date = date.strftime("%Y_%m_%d")
        hours = int(date.strftime("%H"))
        minutes = int(date.strftime("%M"))
        n_half_hours = int(2*hours + minutes/30 + 1)
        dir_path = os.path.join(root, string_date)
        string_date_no_underscore = string_date.replace('_', '')
        file_root= "ToocanCloudMask_SAM_"+string_date_no_underscore+'-'+str(n_half_hours).zfill(3)+'.nc'
        filename = os.path.join(dir_path, file_root)
        return filename

    def load_rel_table(self, file_path):
        """
        Load a .csv file and return its contents as a pandas DataFrame.
        Rel_table contains the path to the output of the storm tracking file per file.

        :param file_path: The path to the .csv file to be loaded.
        :type file_path: str

        :return: A pandas DataFrame containing the data from the .csv file.
        :rtype: pandas.DataFrame
        """
        # print(pd.__version__)
        # print(sys.executable)

        df = pd.read_csv(file_path)
        df.sort_values(by='UTC', ignore_index=True,inplace=True)
        return df

    ## This method is specific to your TIME_RANGE and files in DIR_DATA_IN
    def extract_digit_after_sign(self, input_string):
        """
        Extract the digit after the sign in a string.
        If the string does not contain a sign followed by a digit, return None.
        """

        # Define a regular expression pattern to match the sign followed by a digit
        pattern = r'[+-]\d'

        # Search for the pattern in the input string
        match = re.search(pattern, input_string)

        if match:
            # Extract the digit after the sign
            digit = match.group(0)[1]
            return int(digit)
        else:
            return None
        
    def load_prec(self, casestudy, i_t, sel_dict = None):
        """
        First handmade function (of I hope a long serie)
        Oh and they must del their loadings as they'll be called a lot...
        """
        # i_t = i_t-1 ## Idk what to tell, it just looks more coherent with W this way 
        if i_t in self.settings["prec_i_t_bug_precac"]:
            previous_precac = self.load_var(casestudy, 'Precac', i_t-2, sel_dict = sel_dict)[0]
        else : 
            previous_precac = self.load_var(casestudy, 'Precac', i_t-1, sel_dict = sel_dict)[0]

        current_precac = self.load_var(casestudy, 'Precac', i_t, sel_dict = sel_dict)[0]

        prec = current_precac - previous_precac
        prec = xr.where(prec < 0, 0, prec)
        
        del previous_precac
        del current_precac
        gc.collect()
        return prec

    def compute_qv_sat(self, grid, i_t):
        pp = self.load_var(grid, "PP", i_t)
        tabs = self.load_var(grid, "TABS", i_t)
        # retrieve surface temperature
        p_surf = 100*pp["p"].values[0]+pp["PP"].values[0, 0, :, :]
        t_surf = tabs["TABS"][0,0,:,:].values
        
        original_shape = p_surf.shape
        qv_sat = saturation_specific_humidity(t_surf.ravel(), p_surf.ravel()).reshape(original_shape)
        
        del pp
        del tabs
        del p_surf
        del t_surf
        gc.collect()
        return qv_sat

    def extract_w500(self, grid, i_t):
        w_500 = self.load_var(grid, "W", i_t, z = 32).W # z=32 for 514mb closest to 500        
        return w_500[0,0]
    
    def extract_w500_pos(self, grid, i_t):
        w_500 = self.load_var(grid, "W", i_t, z = 32).W[0,0] # z=32 for 514mb closest to 500   
        w_500 = xr.where(w_500 <0, 0, w_500)   
        return w_500
    
    def compute_qv_sat_2d(self, grid, i_t):
        pres = 100*self.load_var(grid, "PSFC", i_t).values
        temp = self.load_var(grid, "T2mm", i_t).values
        original_shape = pres.shape
        qv_sat = saturation_specific_humidity(temp.ravel(), pres.ravel()).reshape(original_shape)
        del pres
        del temp
        gc.collect()
        return qv_sat

    def extract_w850(self, grid, i_t):
        w_850 = self.load_var(grid, "W", i_t, z = 19).W       
        return w_850[0,0]
    
    def extract_w850_pos(self, grid, i_t):
        w_850 = self.load_var(grid, "W", i_t, z = 19).W[0,0]  
        w_850 = xr.where(w_850 <0, 0, w_850)   
        return w_850
    
    def fetch_om850_over_cond_prec(self, grid, i_t):
        om850 = self.load_var(grid, "OM850", i_t)
        cond_prec = grid.get_cond_prec_on_native_for_i_t(i_t)
        prec = self.load_var(grid, "Prec", i_t)
        om850 = xr.where(prec > cond_prec, om850, np.nan)
        del prec
        del cond_prec
        gc.collect()
        return om850
    
    def fetch_om850_over_cond_prec_lag_1(self, grid, i_t):
        om850 = self.load_var(grid, "OM850", i_t-1) #eazy
        cond_prec = grid.get_cond_prec_on_native_for_i_t(i_t)
        prec = self.load_var(grid, "Prec", i_t)
        om850 = xr.where(prec > cond_prec, om850, np.nan)
        del prec
        del cond_prec
        gc.collect()
        return om850
    
    def fetch_neg_om850_over_cond_prec_lag_1(self, grid, i_t):
        om850 = self.load_var(grid, "OM850", i_t-1) #eazy
        cond_prec = grid.get_cond_prec_on_native_for_i_t(i_t)
        prec = self.load_var(grid, "Prec", i_t)
        om850 = xr.where(prec > cond_prec, om850, np.nan)
        om850 = xr.where(om850 < 0, om850, np.nan)
        del prec
        del cond_prec
        gc.collect()
        return om850
    
    def load_vdcs_conv_prec(self, grid, i_t):
        print("load vdcs conv for ", i_t)
        prec = self.load_prec(grid, i_t)
        mask_prec = prec >=10

        vdcs = self.load_filter_vdcs_seg(grid, i_t).isel(time = 0).rename({'latitude':'lat', 'longitude' : 'lon'})
        vdcs = ~np.isnan(vdcs)

        prec = xr.where(mask_prec & vdcs, prec, np.nan)

        del vdcs
        del mask_prec
        gc.collect()
        return prec

    def load_vdcs_strat_prec(self, grid, i_t):
        print("load vdcs strat for ", i_t)
        prec = self.load_prec(grid, i_t)
        mask_prec = prec < 10
        vdcs = self.load_filter_vdcs_seg(grid, i_t).isel(time = 0).rename({'latitude':'lat', 'longitude' : 'lon'})
        vdcs = ~np.isnan(vdcs)

        prec = xr.where(mask_prec & vdcs, prec, np.nan)
        del vdcs
        del mask_prec
        gc.collect()
        return prec

####### THIS SECTION HAS FUNCTION FOR THE DYAMOND II WINTER PROJECT 
    def get_winter_2_datetime_from_i_t(self, i_t):
        date_ref = dt.datetime(year=self.dict_date_ref["year"], month=self.dict_date_ref["month"], day=self.dict_date_ref["day"])
        delta = dt.timedelta(seconds=i_t*3600) #3600 seconds in 30minutes which is the native timestem
        datetime = delta+date_ref
        return datetime

    def get_dyamond_2_filename_from_i_t(self, i_t):
        new_date = self.get_winter_2_datetime_from_i_t(i_t)
        timestamp = new_date.strftime("%Y%m%d%H")
        
        if self.settings["MODEL"] == "DYAMOND_II_Winter_SAM":
            season_path = 'pr_rlut_sam_winter_' 
        elif self.settings["MODEL"] == "SAM_lowRes" or self.settings["MODEL"] == "SAM_Summer_lowRes":
            season_path = 'pr_rlut_sam_summer_'
        elif self.settings["MODEL"] == "IFS_lowRes" or self.settings["MODEL"] == "IFS_Summer_lowRes":
            season_path = 'pr_rlut_ifs_summer_'
        elif self.settings["MODEL"] == "NICAM_lowRes" or self.settings["MODEL"] == "NICAM_Summer_lowRes":
            season_path = 'pr_rlut_nicam_summer_'
        elif self.settings["MODEL"] == "UM_lowRes" or self.settings["MODEL"] == "UM_Summer_lowRes":
            season_path = "pr_rlut_um_summer_"    
        elif self.settings["MODEL"] == "ARPEGE_lowRes" or self.settings["MODEL"] == "ARPEGE_Summer_lowRes":
            season_path = "pr_rlut_arpnh_summer_"
        elif self.settings["MODEL"] ==  "MPAS_lowRes" or self.settings["MODEL"] == "MPAS_Summer_lowRes":
            season_path = "pr_rlut_mpas_"
        elif self.settings["MODEL"] == "FV3_lowRes" or self.settings["MODEL"] == "FV3_Summer_lowRes":
            season_path = "pr_rlut_fv3_"
        elif self.settings["MODEL"] == "SCREAMv1_lowRes" or self.settings["MODEL"] == "SCREAMv1_Summer_lowRes":
            season_path = "olr_pcp_Summer_SCREAMv1_"
            
        elif self.settings["MODEL"] == "SAM_Winter_lowRes":
            season_path = 'pr_rlut_sam_winter_'
        elif self.settings["MODEL"] == "GEOS_Winter_lowRes":
            season_path = 'pr_rlut_geos_winter_'
        elif self.settings["MODEL"] == "GRIST_Winter_lowRes":
            season_path = 'pr_rlut_grist_'
        elif self.settings["MODEL"] == "IFS_Winter_lowRes":
            season_path = 'pr_rlut_ecmwf_'
        elif self.settings["MODEL"] == "UM_Winter_lowRes":
            season_path = "pr_rlut_um_winter_"    
        elif self.settings["MODEL"] == "ARPEGE_Winter_lowRes":
            season_path = "pr_rlut_arpnh_winter_"
        elif self.settings["MODEL"] == "MPAS_Winter_lowRes":
            season_path = "pr_rlut_mpas_winter_"
        elif self.settings["MODEL"] == "XSHiELD_Winter_lowRes":
            season_path = "pr_rlut_SHiELD-3km_"
        elif self.settings["MODEL"] == "SCREAMv1_Winter_lowRes":
            season_path = "olr_pcp_Winter_SCREAMv1_"

        result = season_path+timestamp+'.nc'
        return result 

    def read_prec(self, grid, i_t):
        current_precac = self.load_var(grid, 'pracc', i_t)
        prec = current_precac #- previous_precac
        del current_precac
        gc.collect()
        return prec

    def diff_precac(self, grid, i_t):
        current_precac = self.load_var(grid, 'Precac', i_t)
        prec = current_precac #- previous_precac
        prec = xr.where(prec < 0, np.nan, prec)
        del current_precac
        gc.collect()
        return prec
    
    
    def diff_tp(self, grid, i_t):
        current_precac = self.load_var(grid, 'tp', i_t)
        prec = current_precac #- previous_precac
        del current_precac
        gc.collect()
        return prec

    def get_sa_tppn(self, grid, i_t):
        current_precac = self.load_var(grid, 'sa_tppn', i_t)
        prec = current_precac #- previous_precac
        del current_precac
        gc.collect()
        return prec

    def get_precipitation_flux(self, grid, i_t):
        prec = self.load_var(grid, 'precipitation_flux', i_t)
        return prec

    def get_pr(self, grid, i_t):
        prec = self.load_var(grid, 'pr', i_t)
        return prec
    
    def get_precipitation(self, grid, i_t):
        prec = self.load_var(grid, 'precipitation', i_t)
        return prec
    
    def get_rain(self, grid, i_t):
        prec = self.load_var(grid, 'param8.1.0', i_t)
        return prec

    def read_seg(self, grid, i_t):
        path_seg_mask = self.settings["DIR_STORM_TRACK"] ## There is the differences
        time = self.get_winter_2_datetime_from_i_t(i_t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=xr.SerializationWarning)
            img_toocan = xr.open_dataset(path_seg_mask, engine='netcdf4').cloud_mask.sel(time = time, latitude = slice(self.settings["BOX"][0], self.settings["BOX"][1]))# because otherwise goes to -60, 60
        return img_toocan
    
    def read_seg_feng(self, grid, i_t):
        path_seg_mask = self.settings["DIR_STORM_TRACK"] ## There is the differences
        time = self.get_winter_2_datetime_from_i_t(i_t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=xr.SerializationWarning)
            img_toocan = xr.open_dataset(path_seg_mask, engine='netcdf4').mcs_mask.sel(time = time, latitude = slice(self.settings["BOX"][0], self.settings["BOX"][1]))# because otherwise goes to -60, 60
        return img_toocan

    def read_filter_vdcs_seg(self, grid, i_t):
        img_toocan = self.read_seg(grid, i_t)
        img_labels = np.unique(img_toocan)[:-1] if np.any(np.isnan(img_toocan)) else np.unique(img_toocan)
        print(len(img_labels))

        # reload storm everytime, fuck it.. dependencies might be doomed
        st = storm_tracker.StormTracker(grid, label_var_id = "MCS_label", overwrite_storms = False, overwrite = False) # takes 2sec with all overwrite to false
        dict = st.get_vdcs_dict()
        valid_labels_per_day, _ = grid.make_labels_per_days_on_dict(dict)
        for i_day, day in enumerate(grid.casestudy.days_i_t_per_var_id[st.label_var_id].keys()):
            if i_t in grid.casestudy.days_i_t_per_var_id[st.label_var_id][day]:
                current_day = day
                current_i_day = i_day
                break
        print("i_day", i_day)
        today_valid_labels = valid_labels_per_day[i_day]
        
        for current_label in img_labels: 
            if current_label not in today_valid_labels:
                img_toocan = img_toocan.where(img_toocan != current_label, np.nan)
        return img_toocan

    def read_filter_vdcs_no_mcs_seg(self, grid, i_t):
        vdcs_mask = self.read_filter_vdcs_seg(grid, i_t)
        mcs_mask = self.read_seg_feng(grid, i_t)
        
        return 0

    def mcs_coverage_cond_prec_15(self, grid, i_t):
        mcs_mask = self.read_seg_feng(grid, i_t)
        cond_prec = grid.get_cond_prec_on_native_for_i_t(i_t, alpha_threshold = 85)
        prec = self.load_var(grid, "Prec", i_t)
        if "FV3" in self.settings["MODEL"] : ## remose last lat because this specifi grid is not centered like the others
            mcs_mask = mcs_mask.isel(latitude=slice(0, -1))
        mcs_mask = xr.where(prec.values > cond_prec, mcs_mask, np.nan)
        del prec
        del cond_prec
        gc.collect()
        return mcs_mask

    def mcs_coverage_cond_prec_25(self, grid, i_t):
        mcs_mask = self.read_seg_feng(grid, i_t)
        cond_prec = grid.get_cond_prec_on_native_for_i_t(i_t, alpha_threshold = 75)
        prec = self.load_var(grid, "Prec", i_t)
        if "FV3" in self.settings["MODEL"] : ## remose last lat because this specifi grid is not centered like the others
            mcs_mask = mcs_mask.isel(latitude=slice(0, -1))
        mcs_mask = xr.where(prec.values > cond_prec, mcs_mask, np.nan)
        del prec
        del cond_prec
        gc.collect()
        return mcs_mask
    
    def sliding_mcs_coverage_cond_prec_15(self, grid, i_t):
        mcs_mask = self.read_seg_feng(grid, i_t)
        previous_mcs_mask = self.read_seg_feng(grid, i_t-1)
        sliding_mcs_mask = mcs_mask.combine_first(previous_mcs_mask)
        del mcs_mask
        del previous_mcs_mask
        cond_prec = grid.get_cond_prec_on_native_for_i_t(i_t, alpha_threshold = 85)
        prec = self.load_var(grid, "Prec", i_t)
        sliding_mcs_mask = xr.where(prec.values > cond_prec, sliding_mcs_mask, np.nan)
        del prec
        del cond_prec
        gc.collect()
        return sliding_mcs_mask
    
    def vdcs_coverage_cond_prec_15(self, grid, i_t):
        mcs_mask = self.read_filter_vdcs_seg(grid, i_t)
        cond_prec = grid.get_cond_prec_on_native_for_i_t(i_t, alpha_threshold = 85)
        prec = self.load_var(grid, "Prec", i_t)

        if "FV3" in self.settings["MODEL"]: ## remose last lat because this specifi grid is not centered like the others
            mcs_mask = mcs_mask.isel(latitude=slice(0, -1))

        mcs_mask = xr.where(prec.values > cond_prec, mcs_mask, np.nan)
        
        del prec
        del cond_prec
        gc.collect()
        return mcs_mask

    def vdcs_coverage_cond_prec_25(self, grid, i_t):
        mcs_mask = self.read_filter_vdcs_seg(grid, i_t)
        cond_prec = grid.get_cond_prec_on_native_for_i_t(i_t, alpha_threshold = 75)
        prec = self.load_var(grid, "Prec", i_t)

        if "FV3" in self.settings["MODEL"]: ## remose last lat because this specifi grid is not centered like the others
            mcs_mask = mcs_mask.isel(latitude=slice(0, -1))

        mcs_mask = xr.where(prec.values > cond_prec, mcs_mask, np.nan)
        
        del prec
        del cond_prec
        gc.collect()
        return mcs_mask

    def clouds_coverage_cond_Prec_15(self, grid, i_t):
        mcs_mask = self.read_seg(grid, i_t)
        cond_prec = grid.get_cond_prec_on_native_for_i_t(i_t, alpha_threshold = 85)
        prec = self.load_var(grid, "Prec", i_t)

        if "FV3" in self.settings["MODEL"] : ## remose last lat because this specifi grid is not centered like the others
            mcs_mask = mcs_mask.isel(latitude=slice(0, -1))
        mcs_mask = xr.where(prec.values > cond_prec, mcs_mask, np.nan)
        
        del prec
        del cond_prec
        gc.collect()
        return mcs_mask

    def clouds_coverage_cond_Prec_25(self, grid, i_t):
        mcs_mask = self.read_seg(grid, i_t)
        cond_prec = grid.get_cond_prec_on_native_for_i_t(i_t, alpha_threshold = 75)
        prec = self.load_var(grid, "Prec", i_t)

        if "FV3" in self.settings["MODEL"] : ## remose last lat because this specifi grid is not centered like the others
            mcs_mask = mcs_mask.isel(latitude=slice(0, -1))
        mcs_mask = xr.where(prec.values > cond_prec, mcs_mask, np.nan)
        
        del prec
        del cond_prec
        gc.collect()
        return mcs_mask

    def sliding_clouds_coverage_cond_Prec_15(self, grid, i_t):
        clouds_mask = self.read_seg(grid, i_t)
        previous_clouds_mask = self.read_seg(grid, i_t-1)
        sliding_clouds_mask = clouds_mask.combine_first(previous_clouds_mask)
        del clouds_mask
        del previous_clouds_mask
        cond_prec = grid.get_cond_prec_on_native_for_i_t(i_t, alpha_threshold = 85)
        prec = self.load_var(grid, "Prec", i_t)
        sliding_clouds_mask = xr.where(prec.values > cond_prec, sliding_clouds_mask, np.nan)
        del prec
        del cond_prec
        gc.collect()
        return sliding_clouds_mask

    def read_sst(self, grid, i_t):
        new_date = self.get_winter_2_datetime_from_i_t(i_t)
        timestamp = new_date.strftime("%Y%m%d%H") ## to adapt for era
        year = f"{new_date.year:04d}"
        month = f"{new_date.month:02d}"
        day = f"{new_date.day:02d}"
        path = f"/bdd/OSTIA_SST_NRT/SST_GLO_SST_L4_NRT_OBSERVATIONS_010_001/METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2/{year}/{month}/{year+month+day}120000-UKMO-L4_GHRSST-SSTfnd-OSTIA-GLOB-v02.0-fv02.0.nc"
        var = xr.open_dataset(path).sel(lon=grid.casestudy.lon_slice,lat=grid.casestudy.lat_slice).analysed_sst.load()[0]
        return var

###### This is for OBS of MCSMIP 
    def get_mcsmip_dyamond_obs_datetime_from_i_t(self, i_t):
        date_ref = dt.datetime(year=self.dict_date_ref["year"], month=self.dict_date_ref["month"], day=self.dict_date_ref["day"])
        delta = dt.timedelta(seconds=i_t*3600)
        datetime = delta+date_ref
        return datetime    
    
    def get_mcsmip_dyamond_obs_filename_from_i_t(self, i_t): 
        new_date = self.get_mcsmip_dyamond_obs_datetime_from_i_t(i_t)
        timestamp = new_date.strftime("%Y%m%d%H")
        result = 'merg_'+timestamp+"_4km-pixel.nc"
        return result
    
    def get_mcsmip_dyamond_obsv7_filename_from_i_t(self, i_t):
        new_date = self.get_mcsmip_dyamond_obs_datetime_from_i_t(i_t)
        timestamp = new_date.strftime("%Y%m%d%H")
        if "Summer" in self.settings["MODEL"]:
            result = "olr_pcp_Summer_OBSv7_"+timestamp+".nc" # Can actually catch model name and Summer Winter from self.settings.... clean that 
        elif "Winter" in self.settings["MODEL"]:
            result = "olr_pcp_Winter_OBSv7_"+timestamp+".nc" # Can actually catch model name and Summer Winter from self.settings.... clean that 
        return result

    def obs_prec(self, grid, i_t):
        current_precac = self.load_var(grid, 'precipitationCal', i_t)
        prec = current_precac #- previous_precac
        del current_precac
        gc.collect()
        return prec

    def obsv7_prec(self, grid, i_t):
        prec = self.load_var(grid, 'precipitation', i_t)
        gc.collect()
        return prec
    
    def obs_seg(self, grid, i_t):
        path_seg_mask = self.settings["DIR_STORM_TRACK"] 
        time = self.get_mcsmip_dyamond_obs_datetime_from_i_t(i_t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=xr.SerializationWarning)
            img_toocan = xr.open_dataset(path_seg_mask, engine='netcdf4').cloud_mask.sel(time = time, latitude = slice(self.settings["BOX"][0], self.settings["BOX"][1]))# because otherwise goes to -60, 60
        return img_toocan
    

    def obs_filter_vdcs_seg(self, grid, i_t):
        img_toocan = self.obs_seg(grid, i_t)
        img_labels = np.unique(img_toocan)[:-1] if np.any(np.isnan(img_toocan)) else np.unique(img_toocan)
        # reload storm everytime, fuck it.. dependencies might be doomed
        st = storm_tracker.StormTracker(grid, label_var_id = "MCS_label", overwrite_storms = False, overwrite = False) # takes 2sec with all overwrite to false
        dict = st.get_vdcs_dict()
        valid_labels_per_day, _ = grid.make_labels_per_days_on_dict(dict)
        for i_day, day in enumerate(grid.casestudy.days_i_t_per_var_id[st.label_var_id].keys()):
            if i_t in grid.casestudy.days_i_t_per_var_id[st.label_var_id][day]:
                current_day = day
                current_i_day = i_day
                break
        today_valid_labels = valid_labels_per_day[i_day]
        
        for current_label in img_labels: 
            if current_label not in today_valid_labels:
                img_toocan = img_toocan.where(img_toocan != current_label, np.nan)
        return img_toocan
        