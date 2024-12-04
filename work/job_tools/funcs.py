import xarray as xr

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

def merge_missing_vars(ft, storms):
    """
    parameters could be storms1 storms2 but whatever, the one on the left is the one that keeps its DCS_number values
    """
    variables_to_add = set(storms.data_vars) - set(ft.data_vars)
    storms_new_vars = storms[variables_to_add]
    storms_new_vars_filtered = storms_new_vars.sel(DCS_number=ft.DCS_number)
    ft_updated = xr.merge([ft, storms_new_vars_filtered])
    return ft_updated