import pickle
import os
import sys
import glob
import tarfile
import shutil 
import json 
import xarray as xr


import numpy as np
import datetime as dt
import re
import pandas as pd 
import yaml
from functools import reduce

class CaseStudy():
    
    """
    Documentation for class CaseStudy
    """
        
    def __init__(self, handler, verbose = False, overwrite = False):
        # Inherit lightly from handler
        self.handler = handler
        with open(self.handler.settings_path, 'r') as file:
            self.settings = yaml.safe_load(file)
            
        # Bools
        self.verbose = verbose
        self.overwrite = overwrite
        # Names and paths
        self.region = self.settings['REGION']
        self.model = self.settings['MODEL']
        self.name = f"{self.model}_{self.region}"

        self.data_out = os.path.join(self.settings['DIR_DATA_OUT'], self.name)

        self._set_region_coord_and_period()
        

        #Maybe make a more robust intern function out of that 

        if not os.path.exists(self.data_out):
            os.makedirs(self.data_out)
            print(f"First instance of {self.name}. It's directory has been created at : {self.data_out}")

        self.variables_names, self.days_i_t_per_var_id, self.new_variables_names, self.new_var_dependencies, self.new_var_functions = self._set_variables(self.overwrite)    
        
    def _set_variables(self, overwrite):
        json_filename = 'var_id_days_i_t.json'
        json_path = os.path.join(self.data_out, json_filename)
        if overwrite or not os.path.exists(json_path):
            if not os.path.exists(json_path) : print(f"Creation of {json_path}")
            if overwrite : print(f"Overwriting the existing variables in {json_path}")
            self.days_i_t_per_var_id = {}
            if self.model ==  'DYAMOND_SAM_post_20_days' or self.model == 'DYAMOND_SAM' or self.model == "SAM_4km_30min_30d" or self.model == "SAM3d":
                self.var_names_2d = self._load_var_id_in_data_in(True)
                self.var_names_3d = self._load_var_id_in_data_in(False)

            elif "lowRes" in self.model:
                self.var_names_2d = self._read_var_id_in_data_in()
                self.var_names_3d = []
            
            self.variables_names = self.var_names_2d + self.var_names_3d
            # quite manual
            self.variables_names, self.days_i_t_per_var_id = self.add_storm_tracking_variables()
            self.new_variables_names, self.new_var_dependencies, self.new_var_functions = self.add_new_var_id()
            
            # try to pass that within method add_new_var_id with a check for Prec as var_id or in dependency
            # doesn't handle dependency of Prec_t_minus_1 just Prec
            
            for var_id in self.variables_names:
                self.days_i_t_per_var_id = self.skip_prec_i_t(var_id)

            print(f"Variables data retrieved. Saving them in {json_path}")
            self.save_var_id_as_json(self.variables_names, self.days_i_t_per_var_id, self.var_names_2d, self.var_names_3d, json_path)
        else :
            if self.verbose : print(f'Found json file at {json_path}, loading it..')
            self.variables_names, self.var_names_2d, self.var_names_3d, self.days_i_t_per_var_id = self.load_var_id_from_json(json_path) #dates are in "year-date-month" for now
            self.new_variables_names = self.settings["new_var"]["variables_id"]
            self.new_var_dependencies = self.settings["new_var"]["dependencies"]
            self.new_var_functions=  self.settings["new_var"]["functions"]
            
        if self.verbose : self._chek_variables_days_and_i_t()

        return self.variables_names, self.days_i_t_per_var_id, self.new_variables_names, self.new_var_dependencies, self.new_var_functions

    def __repr__(self):
        """Creates a printable version of the Distribution object. Only prints the 
        attribute value when its string fits is small enough."""

        out = '< Tempest instance:\n'
        # print keys
        for k in self.__dict__.keys():
            out = out+' . %s: '%k
            if k in ['dist_chunks','chunks_to_ignore']:
                # show type
                out = out+'%s\n'%str(getattr(self,k).__class__)
            else:
                if len(str(getattr(self,k))) < 80:
                    # show value
                    out = out+'%s\n'%str(getattr(self,k))
                else:
                    # show type
                    out = out+'%s\n'%getattr(self,k).__class__
        out = out+' >'

        return out

    def _set_region_coord_and_period(self):
        """
        Simply load the Region of Analysis sections from settings.yaml into class attributes
        """
        # could actually be updated with dtvi 
        self.i_t_min = self.settings['TIME_RANGE'][0]
        self.i_t_max = self.settings['TIME_RANGE'][1]
        self.range_t = range(self.i_t_min, self.i_t_max+1)
        # BOX is [lat_min, lat_max, lon_min, lon_max]
        self.lat_slice = slice(self.settings['BOX'][0], self.settings['BOX'][1])
        self.lon_slice = slice(self.settings['BOX'][2], self.settings['BOX'][3])

    def _get_day_and_i_t(self, filename, timestamp_pattern):
        """
        Weird that this works with MCSMIP dates... because I don't know if file timestamps are spaced of 240 as well ?
        """
        # starting date
        dict_date_ref = self.settings["DATE_REF"]
        date_ref = dt.datetime(year=dict_date_ref["year"], month=dict_date_ref["month"], day=dict_date_ref["day"])
        
        def get_datetime_and_i_t(filename):
            # Extract the timestamp from the file path
            match = re.search(timestamp_pattern, filename)
            # print(filename, match)
            if match:
                timestamp = int(match.group(1))
                # print(timestamp)
                # Calculate the delta in seconds
                delta_t = dt.timedelta(seconds=timestamp * 7.5)
                # Calculate the current date
                date_current = date_ref + delta_t

                i_t = int(timestamp / 240) 
                return date_current, i_t
            else:
                return None  # Handle cases where the timestamp couldn't be extracted

        # time dimension
        date_time, i_t = get_datetime_and_i_t(filename)
        day = date_time.strftime("%y-%m-%d")
        return day, i_t

    def _load_var_id_in_data_in(self, bool_2d):
        """
        this functions loads the data from either DIR_DATA_2D_IN or DIR_DATA_3D_IN

        :param dir: The path to your DIR_DATA_IN in .yaml

        :return
            var_id: list of variables found
            
        :update self.days_i_t_per_var_id: a dictionnary that contains the days and correspong indexes per var_id    
            self.days_i_t_per_var_id[var_id] = dict with keys the dates and values the indexes
        """
        var_names = []
        if bool_2d:
            dir = self.settings['DIR_DATA_2D_IN']
            variable_pattern = re.compile(r'\.([A-Za-z0-9]+)\.2D\.nc$') 
            timestamp_pattern =  re.compile(r'(\d{10})\.\w+\.2D\.nc')
        else:
            dir = self.settings['DIR_DATA_3D_IN']
            variable_pattern = re.compile(r'_([A-Za-z0-9]+)\.nc$') 
            timestamp_pattern = re.compile(r'_(\d{10})_[A-Za-z0-9]+\.nc$')

        # Define a regular expression pattern to extract variable names from filenames.
        files = glob.glob(dir +'/*.nc')
        for filename in sorted(files):
            match = variable_pattern.search(filename)
            # print(filename, match)
            if match:
                var_id = match.group(1)
                # print(var_id)
                if var_id not in var_names : 
                    var_names.append(var_id)
                    self.days_i_t_per_var_id[var_id] = {}
                    
                day, i_t = self._get_day_and_i_t(filename, timestamp_pattern)
                if day not in list(self.days_i_t_per_var_id[var_id].keys()):
                    self.days_i_t_per_var_id[var_id][day] = [i_t]
                else :
                    self.days_i_t_per_var_id[var_id][day].append(i_t)
                    
        return var_names

    def _read_var_id_in_data_in(self):
        """
        equivalent of _load_var_id_in_data_in but for Dyamond WINTER
        """
        dir = self.settings['DIR_DATA_2D_IN']
        files = glob.glob(dir + '*.nc')
        first_file = files[0]
        first_ds = xr.open_dataset(first_file)
        var_names = []
        for var_id, variable in first_ds.data_vars.items():
            # Check if 'time', 'lon', and 'lat' are in the dimensions of the variable
            if {'time', 'lon', 'lat'}.issubset(set(variable.dims)) or {'xtime', 'lon', 'lat'}.issubset(set(variable.dims)): ## because MPAS guys are douchbags
                # If the condition is met, append the variable name to the list
                var_names.append(var_id)
                self.days_i_t_per_var_id[var_id] = {}

        timestamp_pattern = re.compile(r'(\d{4}\d{2}\d{2}\d{2})\.nc')

        if self.settings["MODEL"] == 'OBS_lowRes':
            timestamp_pattern = re.compile(r'(\d{10})')
                                           
        for var_id in var_names:
            for filename in sorted(files):
                dict_date_ref = self.settings["DATE_REF"]
                date_ref = dt.datetime(year=dict_date_ref["year"], month=dict_date_ref["month"], day=dict_date_ref["day"])
                match = re.search(timestamp_pattern, filename)
                timestamp = int(match.group(1))
                date = dt.datetime.strptime(str(timestamp), "%Y%m%d%H")
                day = date.strftime("%y-%m-%d")
                delta_t = (date - date_ref)/3600
                i_t = delta_t.seconds
                if day not in list(self.days_i_t_per_var_id[var_id].keys()):
                    self.days_i_t_per_var_id[var_id][day] = [i_t]
                else: 
                    self.days_i_t_per_var_id[var_id][day].append(i_t)
        return var_names

    def _chek_variables_days_and_i_t(self):
        for var_id in self.variables_names:
            print(var_id)

            days = list(self.days_i_t_per_var_id[var_id].keys())
            print('day:      (#t)  t_i-t_f')
            for day in days:
                i_t_day = self.days_i_t_per_var_id[var_id][day]
                print('%s: (%d) %d-%d'%(day,len(i_t_day), i_t_day[0], i_t_day[-1]))
        print('\n')

    def save_var_id_as_json(self, variables_id, days_i_t_per_var_id, var_2d, var_3d, json_filename):
        """
        Save the variables_id and days_i_t_per_var_id as a JSON file.

        :param variables_id: list of variables found
        :param days_i_t_per_var_id: a dictionary containing the days and corresponding indexes per var_id
        :param json_filename: The name of the JSON file to save the data.
        """
        data_to_save = {
            "variables_id": variables_id,
            "variables_2d_id" : var_2d,
            "variables_3d_id" : var_3d,
            "days_i_t_per_var_id": days_i_t_per_var_id
        }

        with open(json_filename, 'w') as json_file:
            json.dump(data_to_save, json_file, indent=4)
        print(f"Data saved as {json_filename}")
        
    def load_var_id_from_json(self, json_filename):
        """
        Load the variables_id and days_i_t_per_var_id from a JSON file.

        :param json_filename: The name of the JSON file to load the data from.
        :return: A tuple containing variables_id and days_i_t_per_var_id.
        """
        try:
            with open(json_filename, 'r') as json_file:
                data = json.load(json_file)
                variables_id = data.get("variables_id", [])
                var_2d = data.get("variables_2d_id", [])
                var_3d = data.get("variables_3d_id", [])
                days_i_t_per_var_id = data.get("days_i_t_per_var_id", {})
                print(f"Data loaded from {json_filename}")
                return variables_id, var_2d, var_3d, days_i_t_per_var_id
        except FileNotFoundError:
            print(f"File {json_filename} not found.")
            return None, None

    def _update_ditvi(self, var_id, dependency):
        """
        getting passed the new variables names var_id, and the variables names it depends of as a list of string dependency 
        this functions make the intersection of dates and i_t for the new variable.

        dependency is a list of either directly original or new var_id, but it can also contain a +n or -n as to specify an offset in its indexes 
        e.g. :  Prec = Precac - Precac-1 ; dependencies["Prec"] = ["Precac", "Precac-1"]
        """
        
        ### 
        # This block manages the dates of the variable
        # It builds dates
        ###
        
        ddates = []
        for dvar_id in dependency:
            ## Maybe if finally the date is empty of any i_t should be removed from the keys! even if it should not propagate any issue      
            if "+" in dvar_id or "-" in dvar_id:
                ddates.append(list(self.days_i_t_per_var_id[dvar_id[:-2]].keys())) #[:-2] is to remove the +1 or -2 at the end of the var
                if "-" in dvar_id:
                    # this just has to add end dates if data is actually available further
                    offset = self.handler.extract_digit_after_sign(dvar_id) #should be a casestudy function, isn't use anywhere else...
                    last_date = list(self.days_i_t_per_var_id[dvar_id[:-2]].keys())[-1]
                    last_i_t_last_date = self.days_i_t_per_var_id[dvar_id[:-2]][last_date][-1]

                    if last_i_t_last_date+offset // 48 == 0 : # then add one more day
                        day_int = int(last_date[6:8])
                        if day_int != 31 :
                            new_day = day_int + 1
                        new_day_str = last_date[:6]+str(new_day).zfill(2)
                        ddates.append(new_day_str)
            else : 
                ddates.append(list(self.days_i_t_per_var_id[dvar_id].keys()))

        if len(ddates)>0:
            dates = reduce(lambda x, y: list(set(x) & set(y)), ddates)
        dates = np.sort(dates)
        
        ### 
        # This block manages the indexes within the dates of the variable
        # It builds i_t_per_date
        ###
        
        i_t_per_date = []
        for i_date, date in enumerate(dates):
            dindexes = []
            for dvar_id in dependency:
                if "-" in dvar_id:
                    offset = self.handler.extract_digit_after_sign(dvar_id)
                    if i_date!= 0 : ## Ce n'est pas le 1er jour on peut récupérer celui d'avant
                        prev_date = dates[i_date-1]
                        prev_date_indexes = []
                        for i in range(offset):
                            #[:-2] is to avoid the +int at the end of var_id
                            prev_date_indexes.append(self.days_i_t_per_var_id[dvar_id[:-2]][prev_date][-(i+1)]+offset)
                    this_date_indexes = [self.days_i_t_per_var_id[dvar_id[:-2]][date][i]+offset for i in range(len(self.days_i_t_per_var_id[dvar_id[:-2]][date])-offset)]
                    
                    if i_date !=0: 
                        dindexes.append(prev_date_indexes+this_date_indexes)
                    else : 
                        dindexes.append(this_date_indexes)
                else : 
                    dindexes.append(list(self.days_i_t_per_var_id[dvar_id][date]))
            i_t_per_date.append(reduce(lambda x, y: list(set(x) & set(y)), dindexes))

        ### 
        # This block update ditvi based on dates and i_t_per_date
        ###

        self.days_i_t_per_var_id[var_id] = {}
        for i,date in enumerate(dates): 
            self.days_i_t_per_var_id[var_id][date] = sorted(i_t_per_date[i])
        return self.days_i_t_per_var_id
    
    def add_new_var_id(self):
        """
        Reads the new_variables in settings and update ditvi with them, updating the ditvi accordingly and also loading the functions
        that'll be stored in handler to load the variables.
        output: 
            new_var_names, basically their var_id
            dependencies, a dict of keys new_var_names that will be the variables that must be loaded to compute the new one
            functions, a dict of keys new_var_names that calls the function to load this var
        """

        # loading from settings
        new_var_names = self.settings["new_var"]["variables_id"]
        dependencies = self.settings["new_var"]["dependencies"]                                                     
        functions = self.settings["new_var"]["functions"]

        for var_id in new_var_names:
            if var_id not in self.variables_names:
                dependency = dependencies[var_id]

                if var_id not in self.variables_names: self.variables_names.append(var_id)
                print(var_id)
         
                if len(dependency) > 0 : # If you add new variables that have no dependency you must create your own function to load them 
                                         # and update ditvi like with add_storm_tracking_variables
                    self.days_i_t_per_var_id = self._update_ditvi(var_id, dependency)
                    if var_id == "Prec":
                        self.skip_prec_i_t(var_id)
                    ## maybe not necessary... ? 
                    # if "Prec" in dependency:
                    #     self.skip_prec_i_t(var_id)

        return new_var_names, dependencies, functions
        
    def add_storm_tracking_variables(self, vanilla = False):
        """
        Could be a whole Class as there will  be a lot of future development
        Add MCS_label to the variables, and update ditvi according to rel_table
        """
        if "MCS_label" not in self.variables_names:
            self.variables_names.append("MCS_label")
        
        if "MCS_label_Tb_Feng" not in self.variables_names:
            self.variables_names.append("MCS_label_Tb_Feng")

        if "Conv_MCS_label" not in self.variables_names:
            self.variables_names.append("Conv_MCS_label")

        if "MCS_Feng" not in self.variables_names:
            self.variables_names.append("MCS_Feng")

        if "vDCS" not in self.variables_names:
            self.variables_names.append("vDCS")
        
        if "vDCS_no_MCS" not in self.variables_names:
            self.variables_names.append("vDCS_no_MCS")

        if "sst" not in self.variables_names:
            self.variables_names.append("sst")
        # let's get rid off rel_table because of duplicates issue (not quantified)
        # if vanilla :
        #     # I love SQL time
        #     self.rel_table['Unnamed: 0.1'] = self.rel_table['Unnamed: 0.1'].astype(int)
        #     self.rel_table['ditvi_date_key'] = pd.to_datetime(self.rel_table[['year', 'month', 'day']]).dt.strftime("%y-%m-%d")
        #     # print(pd.to_datetime(self.rel_table[['year', 'month', 'day']]))
        #     self.days_i_t_per_var_id["MCS_label"] = self.rel_table.groupby('ditvi_date_key')['Unnamed: 0.1'].apply(lambda group: (group+1).tolist()).to_dict()
            
        #     return self.variables_names, self.days_i_t_per_var_id
        # else :
            ## I have a feeling that joint distrib doesn't work well otherwise but to check

        var_id_fully_avail = self.settings["new_var"]["dependencies"]["Prec"][0]
            
        self.days_i_t_per_var_id["MCS_label"] = self.days_i_t_per_var_id[var_id_fully_avail]
        self.days_i_t_per_var_id["MCS_label_Tb_Feng"] = self.days_i_t_per_var_id[var_id_fully_avail]
        self.days_i_t_per_var_id["Conv_MCS_label"] = self.days_i_t_per_var_id[var_id_fully_avail]
        self.days_i_t_per_var_id["MCS_Feng"] = self.days_i_t_per_var_id[var_id_fully_avail]
        self.days_i_t_per_var_id["vDCS"] = self.days_i_t_per_var_id[var_id_fully_avail]
        self.days_i_t_per_var_id["vDCS_no_MCS"] = self.days_i_t_per_var_id[var_id_fully_avail]

        self.days_i_t_per_var_id["sst"] = {key: [value[0]] for key, value in self.days_i_t_per_var_id[var_id_fully_avail].items()}

        return self.variables_names, self.days_i_t_per_var_id

    def skip_prec_i_t(self, var_id): 
        """
            Skip the i_t specified in settings
            Only for "Prec" but it could be generalized to any variables that depends on Prec...
            Or to be called before dependency is passed to new variables hm...
            should be to Prec_t_minus_1
        """
        to_skip = self.settings["skip_prec_i_t"]

        i_min, i_max = self.settings["TIME_RANGE"][0], self.settings["TIME_RANGE"][1] 
        days = list(self.days_i_t_per_var_id[var_id].keys())

        for day in days:
            indexes = self.days_i_t_per_var_id[var_id][day]
            if type(indexes) is list : 
                filtered_indexes = [idx for idx in indexes if (idx >= i_min) and (idx<=i_max)]
            if len(filtered_indexes) == 0 : del self.days_i_t_per_var_id[var_id][day]
            else : 
                for i_t in to_skip:
                    if i_t in filtered_indexes:
                        filtered_indexes.remove(i_t)
                if len(filtered_indexes)==0 : del self.days_i_t_per_var_id[var_id][day]
                else : self.days_i_t_per_var_id[var_id][day] = filtered_indexes             
                    
        return self.days_i_t_per_var_id
      
    def get_i_day_from_i_t(self, i_t):
        """
        Explicit name, instead could build a reversed dict of ditvi.
        Uses Prec as it is now used to expand Treshold of cond_Prec to a native shape map
        """
        for i_day, day in enumerate(self.days_i_t_per_var_id["Prec"].keys()):
            if i_t in self.days_i_t_per_var_id["Prec"][day] : 
                return i_day