from sklearn.preprocessing import StandardScaler
import numpy as np



#### v0 ####

def scale_profiles(ds, scaler=None):

    # Extract the profile data
    TABS_init_profile = ds['TABS_init_profile'].values
    QV_init_profile = ds['QV_init_profile'].values
    U_init_profile = ds['U_init_profile'].values
    V_init_profile = ds['V_init_profile'].values

    TABS_max_instant_profile = ds['TABS_max_instant_profile'].values
    QV_max_instant_profile = ds['QV_max_instant_profile'].values
    U_max_instant_profile = ds['U_max_instant_profile'].values
    V_max_instant_profile = ds['V_max_instant_profile'].values

    # Stack initial profiles
    profiles_init_stack = np.stack([
        TABS_init_profile,
        QV_init_profile,
        U_init_profile,
        V_init_profile
    ], axis=2)

    # Stack max instant profiles
    profiles_max_stack = np.stack([
        TABS_max_instant_profile,
        QV_max_instant_profile,
        U_max_instant_profile,
        V_max_instant_profile
    ], axis=2)

    # Concatenate initial and max profiles
    X = np.concatenate([profiles_init_stack, profiles_max_stack], axis=2)
    num_samples, num_levels, num_features = X.shape

    # Reshape to 2D array for scaling
    X_reshaped = X.reshape(-1, num_features)

    # Initialize or use provided scaler
    if scaler is None:
        scaler_profiles = StandardScaler()
        X_scaled_reshaped = scaler_profiles.fit_transform(X_reshaped)
    else:
        scaler_profiles = scaler
        X_scaled_reshaped = scaler_profiles.transform(X_reshaped)

    # Reshape back to original shape
    X_scaled = X_scaled_reshaped.reshape(num_samples, num_levels, num_features)

    return X, X_scaled, scaler_profiles


#### v1 ####

def compute_direction_and_speed(lon1, lat1, lon2, lat2, time_diff_hours):
    R = 6371.0  # Earth's radius in kilometers
    
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    
    direction = np.arctan2(delta_lat, delta_lon)
    
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R * c 
    
    speed =  distance_km / time_diff_hours
    
    return direction, 3.6*speed

def projection_on_propagation_direction(U, V, u_p, v_p):
    u_p = u_p[:, np.newaxis]
    v_p = v_p[:, np.newaxis]
    tangeant_wind = U * u_p + V * v_p
    normal_wind = - U * u_p + V * v_p
    return tangeant_wind, normal_wind

def scale_profiles_v1(ds, scaler=None):
    # Extract the profile data
    TABS_init_profile = ds['TABS_init_profile'].values
    QV_init_profile = ds['QV_init_profile'].values
    U_init_profile = ds['U_init_profile'].values
    V_init_profile = ds['V_init_profile'].values

    TABS_max_instant_profile = ds['TABS_max_instant_profile'].values
    QV_max_instant_profile = ds['QV_max_instant_profile'].values
    U_max_instant_profile = ds['U_max_instant_profile'].values
    V_max_instant_profile = ds['V_max_instant_profile'].values

    z_levels = ds.z.values
    # Compute wind shear for initial profiles
    delta_U_init = np.diff(U_init_profile, axis=1)/np.diff(z_levels)
    delta_V_init = np.diff(V_init_profile, axis=1)/np.diff(z_levels)
    shear_magnitude_init = np.sqrt(delta_U_init**2 + delta_V_init**2)
    shear_direction_init = np.arctan2(delta_V_init, delta_U_init)  # In radians

    # Compute wind shear for max instant profiles
    delta_U_max = np.diff(U_max_instant_profile, axis=1)/np.diff(z_levels)
    delta_V_max = np.diff(V_max_instant_profile, axis=1)/np.diff(z_levels)
    shear_magnitude_max = np.sqrt(delta_U_max**2 + delta_V_max**2)
    shear_direction_max = np.arctan2(delta_V_max, delta_U_max)  # In radians

    propagation_direction, propagation_speed = compute_direction_and_speed(
        ds['lon_init'].values, 
        ds['lat_init'].values, 
        ds['lon_max_instant'].values, 
        ds['lat_max_instant'].values, 
        (ds["time_max_instant"].values - ds["time_init"].values).astype(float)/2
    )

    u_p = np.cos(propagation_direction)
    v_p = np.sin(propagation_direction)

    wind_tangeant_init, wind_normal_init = projection_on_propagation_direction(U_init_profile, V_init_profile, u_p, v_p)
    wind_tangeant_max_instant, wind_normal_max_instant = projection_on_propagation_direction(U_max_instant_profile, V_max_instant_profile, u_p, v_p)
    shear_tangeant_init, shear_normal_init = projection_on_propagation_direction(delta_U_init, delta_V_init, u_p, v_p)
    shear_tangeant_max_instant, shear_normal_max_instant = projection_on_propagation_direction(delta_U_max, delta_V_max, u_p, v_p)

    shear_magnitude_init_bis = np.sqrt(shear_tangeant_init**2 + shear_normal_init**2)
    shear_magnitude_max_bis = np.sqrt(shear_tangeant_max_instant**2 + shear_normal_max_instant**2)

    # Add padding at the top layer to maintain the original number of levels
    pad_width = ((0, 0), (1, 0))  # Pad before the first vertical level
    shear_magnitude_init = np.pad(shear_magnitude_init, pad_width, mode='constant', constant_values=0)
    shear_direction_init = np.pad(shear_direction_init, pad_width, mode='constant', constant_values=0)
    shear_magnitude_max = np.pad(shear_magnitude_max, pad_width, mode='constant', constant_values=0)
    shear_direction_max = np.pad(shear_direction_max, pad_width, mode='constant', constant_values=0)

    shear_tangeant_init = np.pad(shear_tangeant_init, pad_width, mode='constant', constant_values=0)
    shear_normal_init = np.pad(shear_normal_init, pad_width, mode='constant', constant_values=0)
    shear_tangeant_max_instant = np.pad(shear_tangeant_max_instant, pad_width, mode='constant', constant_values=0)
    shear_normal_max_instant = np.pad(shear_normal_max_instant, pad_width, mode='constant', constant_values=0)

    shear_magnitude_init_bis = np.pad(shear_magnitude_init_bis, pad_width, mode='constant', constant_values=0)
    shear_magnitude_max_bis = np.pad(shear_magnitude_max_bis, pad_width, mode='constant', constant_values=0)

    profiles_init_stack = np.stack([
        TABS_init_profile,
        QV_init_profile,
        wind_tangeant_init,
        wind_normal_init,
        shear_tangeant_init,
        shear_normal_init,
        shear_magnitude_init,
        # shear_magnitude_init_bis, it works
    ], axis=2)

    profiles_max_stack = np.stack([
        TABS_max_instant_profile,
        QV_max_instant_profile,
        wind_tangeant_max_instant, 
        wind_normal_max_instant,
        shear_tangeant_max_instant, 
        shear_normal_max_instant,
        shear_magnitude_max,
        # shear_magnitude_max_bis, it works
    ], axis=2)

    # Concatenate initial and max profiles
    X = np.concatenate([profiles_init_stack, profiles_max_stack], axis=2)
    num_samples, num_levels, num_features = X.shape

    # Reshape to 2D array for scaling
    X_reshaped = X.reshape(-1, num_features)

    # Initialize or use provided scaler
    if scaler is None:
        scaler_profiles = StandardScaler()
        X_scaled_reshaped = scaler_profiles.fit_transform(X_reshaped)
    else:
        scaler_profiles = scaler
        X_scaled_reshaped = scaler_profiles.transform(X_reshaped)

    X_scaled = X_scaled_reshaped.reshape(num_samples, num_levels, num_features)
    return X, X_scaled, scaler_profiles

#### v2 ####
# removed first an last level + rolling average for smoothing
####

def moving_average(data, n, axis=1):
    # Move axis to the first position for easier processing
    data = np.moveaxis(data, axis, 0)
    
    # Apply moving average along the first axis
    shape = data.shape
    result = np.empty((shape[0] - n + 1,) + shape[1:])
    for idx in np.ndindex(*shape[1:]):  # Loop over all remaining dimensions
        result[(...,) + idx] = np.convolve(data[(...,) + idx], np.ones(n) / n, mode='valid')
    
    # Move the axis back to its original position
    result = np.moveaxis(result, 0, axis)

    return result[:,2:]


def scale_profiles_v2(ds, scaler=None):
    # Extract the profile data
    TABS_init_profile = ds['TABS_init_profile'].values
    QV_init_profile = ds['QV_init_profile'].values
    U_init_profile = ds['U_init_profile'].values
    V_init_profile = ds['V_init_profile'].values

    TABS_max_instant_profile = ds['TABS_max_instant_profile'].values
    QV_max_instant_profile = ds['QV_max_instant_profile'].values
    U_max_instant_profile = ds['U_max_instant_profile'].values
    V_max_instant_profile = ds['V_max_instant_profile'].values

    z_levels = ds.z.values
    # Compute wind shear for initial profiles
    delta_U_init = 1000*np.diff(U_init_profile, axis=1)/np.diff(z_levels) # (m/s)/km
    delta_V_init = 1000*np.diff(V_init_profile, axis=1)/np.diff(z_levels) # (m/s)/km
    shear_magnitude_init = np.sqrt(delta_U_init**2 + delta_V_init**2)
    shear_direction_init = np.arctan2(delta_V_init, delta_U_init)  # In radians

    # Compute wind shear for max instant profiles
    delta_U_max = 1000*np.diff(U_max_instant_profile, axis=1)/np.diff(z_levels) # (m/s)/km
    delta_V_max = 1000*np.diff(V_max_instant_profile, axis=1)/np.diff(z_levels) # (m/s)/km
    # shear_magnitude_max = np.sqrt(delta_U_max**2 + delta_V_max**2)
    # shear_direction_max = np.arctan2(delta_V_max, delta_U_max)  # In radians

    propagation_direction, propagation_speed = compute_direction_and_speed(
        ds['lon_init'].values, 
        ds['lat_init'].values, 
        ds['lon_max_instant'].values, 
        ds['lat_max_instant'].values, 
        (ds["time_max_instant"].values - ds["time_init"].values).astype(float)/2
    )

    u_p = np.cos(propagation_direction)
    v_p = np.sin(propagation_direction)

    wind_tangeant_init, wind_normal_init = projection_on_propagation_direction(U_init_profile, V_init_profile, u_p, v_p)
    wind_tangeant_max_instant, wind_normal_max_instant = projection_on_propagation_direction(U_max_instant_profile, V_max_instant_profile, u_p, v_p)
    shear_tangeant_init, shear_normal_init = projection_on_propagation_direction(delta_U_init, delta_V_init, u_p, v_p)
    shear_tangeant_max_instant, shear_normal_max_instant = projection_on_propagation_direction(delta_U_max, delta_V_max, u_p, v_p)

    # shear_magnitude_init_bis = np.sqrt(shear_tangeant_init**2 + shear_normal_init**2)
    # shear_magnitude_max_bis = np.sqrt(shear_tangeant_max_instant**2 + shear_normal_max_instant**2)

    # Add padding at the bottom layer to maintain the original number of levels
    pad_width = ((0, 0), (1, 0))  # Pad before the first vertical level
    shear_magnitude_init = np.pad(shear_magnitude_init, pad_width, mode='constant', constant_values=0)
    shear_direction_init = np.pad(shear_direction_init, pad_width, mode='constant', constant_values=0)
    # shear_magnitude_max = np.pad(shear_magnitude_max, pad_width, mode='constant', constant_values=0)
    # shear_direction_max = np.pad(shear_direction_max, pad_width, mode='constant', constant_values=0)

    shear_tangeant_init = np.pad(shear_tangeant_init, pad_width, mode='constant', constant_values=0)
    shear_normal_init = np.pad(shear_normal_init, pad_width, mode='constant', constant_values=0)
    shear_tangeant_max_instant = np.pad(shear_tangeant_max_instant, pad_width, mode='constant', constant_values=0)
    shear_normal_max_instant = np.pad(shear_normal_max_instant, pad_width, mode='constant', constant_values=0)

    # shear_magnitude_init_bis = np.pad(shear_magnitude_init_bis, pad_width, mode='constant', constant_values=0)
    # shear_magnitude_max_bis = np.pad(shear_magnitude_max_bis, pad_width, mode='constant', constant_values=0)
    
    profiles_init_stack = np.stack([
        TABS_init_profile[:,4:-2],
        QV_init_profile[:,4:-2],
        moving_average(wind_tangeant_init,3)[:,1:-1],
        moving_average(wind_normal_init, 3)[:,1:-1],
        moving_average(shear_tangeant_init, 5),
        moving_average(shear_normal_init, 5),
        # moving_average(shear_magnitude_init, 5),
        # shear_magnitude_init_bis, it works
    ], axis=2)

    profiles_max_stack = np.stack([
        TABS_max_instant_profile[:,4:-2],
        QV_max_instant_profile[:,4:-2],
        moving_average(wind_tangeant_max_instant, 3)[:,1:-1], 
        moving_average(wind_normal_max_instant, 3)[:,1:-1],
        moving_average(shear_tangeant_max_instant, 5), 
        moving_average(shear_normal_max_instant, 5),
        # moving_average(shear_magnitude_max, 5),
        # shear_magnitude_max_bis, it works
    ], axis=2)

    # Concatenate initial and max profiles
    X = np.concatenate([profiles_init_stack, profiles_max_stack], axis=2)
    num_samples, num_levels, num_features = X.shape

    # Reshape to 2D array for scaling
    X_reshaped = X.reshape(-1, num_features)

    # Initialize or use provided scaler
    if scaler is None:
        scaler_profiles = StandardScaler()
        X_scaled_reshaped = scaler_profiles.fit_transform(X_reshaped)
    else:
        scaler_profiles = scaler
        X_scaled_reshaped = scaler_profiles.transform(X_reshaped)

    X_scaled = X_scaled_reshaped.reshape(num_samples, num_levels, num_features)
    return X, X_scaled, scaler_profiles


def scale_profiles_v3(ds, scaler=None):
    # Extract the profile data
    TABS_init_profile = ds['TABS_init_profile'].values
    QV_init_profile = ds['QV_init_profile'].values
    U_init_profile = ds['U_init_profile'].values
    V_init_profile = ds['V_init_profile'].values


    z_levels = ds.z.values
    # Compute wind shear for initial profiles
    delta_U_init = 1000*np.diff(U_init_profile, axis=1)/np.diff(z_levels) # (m/s)/km
    delta_V_init = 1000*np.diff(V_init_profile, axis=1)/np.diff(z_levels) # (m/s)/km
    shear_magnitude_init = np.sqrt(delta_U_init**2 + delta_V_init**2)
    shear_direction_init = np.arctan2(delta_V_init, delta_U_init)  # In radians

    # Compute wind shear for max instant profiles
    # shear_magnitude_max = np.sqrt(delta_U_max**2 + delta_V_max**2)
    # shear_direction_max = np.arctan2(delta_V_max, delta_U_max)  # In radians

    propagation_direction, propagation_speed = compute_direction_and_speed(
        ds['lon_init'].values, 
        ds['lat_init'].values, 
        ds['lon_max_instant'].values, 
        ds['lat_max_instant'].values, 
        (ds["time_max_instant"].values - ds["time_init"].values).astype(float)/2
    )

    u_p = np.cos(propagation_direction)
    v_p = np.sin(propagation_direction)

    wind_tangeant_init, wind_normal_init = projection_on_propagation_direction(U_init_profile, V_init_profile, u_p, v_p)

    shear_tangeant_init, shear_normal_init = projection_on_propagation_direction(delta_U_init, delta_V_init, u_p, v_p)

    # shear_magnitude_init_bis = np.sqrt(shear_tangeant_init**2 + shear_normal_init**2)
    # shear_magnitude_max_bis = np.sqrt(shear_tangeant_max_instant**2 + shear_normal_max_instant**2)

    # Add padding at the bottom layer to maintain the original number of levels
    pad_width = ((0, 0), (1, 0))  # Pad before the first vertical level
    shear_magnitude_init = np.pad(shear_magnitude_init, pad_width, mode='constant', constant_values=0)
    shear_direction_init = np.pad(shear_direction_init, pad_width, mode='constant', constant_values=0)
    # shear_magnitude_max = np.pad(shear_magnitude_max, pad_width, mode='constant', constant_values=0)
    # shear_direction_max = np.pad(shear_direction_max, pad_width, mode='constant', constant_values=0)

    shear_tangeant_init = np.pad(shear_tangeant_init, pad_width, mode='constant', constant_values=0)
    shear_normal_init = np.pad(shear_normal_init, pad_width, mode='constant', constant_values=0)
    # shear_tangeant_max_instant = np.pad(shear_tangeant_max_instant, pad_width, mode='constant', constant_values=0)
    # shear_normal_max_instant = np.pad(shear_normal_max_instant, pad_width, mode='constant', constant_values=0)

    # shear_magnitude_init_bis = np.pad(shear_magnitude_init_bis, pad_width, mode='constant', constant_values=0)
    # shear_magnitude_max_bis = np.pad(shear_magnitude_max_bis, pad_width, mode='constant', constant_values=0)
    
    profiles_init_stack = np.stack([
        TABS_init_profile[:,4:-2],
        QV_init_profile[:,4:-2],
        moving_average(wind_tangeant_init,3)[:,1:-1],
        moving_average(wind_normal_init, 3)[:,1:-1],
        moving_average(shear_tangeant_init, 5),
        moving_average(shear_normal_init, 5),
        # moving_average(shear_magnitude_init, 5),
        # shear_magnitude_init_bis, it works
    ], axis=2)

    # profiles_max_stack = np.stack([
    #     moving_average(shear_tangeant_max_instant, 5), 
    #     moving_average(shear_normal_max_instant, 5),
    #     # moving_average(shear_magnitude_max, 5),
    #     # shear_magnitude_max_bis, it works
    # ], axis=2)

    # Concatenate initial and max profiles
    # X = np.concatenate([profiles_init_stack, profiles_max_stack], axis=2)
    X = profiles_init_stack
    num_samples, num_levels, num_features = X.shape

    # Reshape to 2D array for scaling
    X_reshaped = X.reshape(-1, num_features)

    # Initialize or use provided scaler
    if scaler is None:
        scaler_profiles = StandardScaler()
        X_scaled_reshaped = scaler_profiles.fit_transform(X_reshaped)
    else:
        scaler_profiles = scaler
        X_scaled_reshaped = scaler_profiles.transform(X_reshaped)

    X_scaled = X_scaled_reshaped.reshape(num_samples, num_levels, num_features)
    return X, X_scaled, scaler_profiles


def scale_profiles_v4(ds, scaler=None):
    # Extract the profile data
    # TABS_init_profile = ds['TABS_init_profile'].values
    # QV_init_profile = ds['QV_init_profile'].values
    U_init_profile = ds['U_init_profile'].values
    V_init_profile = ds['V_init_profile'].values


    z_levels = ds.z.values
    # Compute wind shear for initial profiles
    delta_U_init = 1000*np.diff(U_init_profile, axis=1)/np.diff(z_levels) # (m/s)/km
    delta_V_init = 1000*np.diff(V_init_profile, axis=1)/np.diff(z_levels) # (m/s)/km
    # shear_magnitude_init = np.sqrt(delta_U_init**2 + delta_V_init**2)
    # shear_direction_init = np.arctan2(delta_V_init, delta_U_init)  # In radians

    # Compute wind shear for max instant profiles
    # shear_magnitude_max = np.sqrt(delta_U_max**2 + delta_V_max**2)
    # shear_direction_max = np.arctan2(delta_V_max, delta_U_max)  # In radians

    propagation_direction, propagation_speed = compute_direction_and_speed(
        ds['lon_init'].values, 
        ds['lat_init'].values, 
        ds['lon_max_instant'].values, 
        ds['lat_max_instant'].values, 
        (ds["time_max_instant"].values - ds["time_init"].values).astype(float)/2
    )

    u_p = np.cos(propagation_direction)
    v_p = np.sin(propagation_direction)

    wind_tangeant_init, wind_normal_init = projection_on_propagation_direction(U_init_profile, V_init_profile, u_p, v_p)

    # shear_tangeant_init, shear_normal_init = projection_on_propagation_direction(delta_U_init, delta_V_init, u_p, v_p)

    # shear_magnitude_init_bis = np.sqrt(shear_tangeant_init**2 + shear_normal_init**2)
    # shear_magnitude_max_bis = np.sqrt(shear_tangeant_max_instant**2 + shear_normal_max_instant**2)

    # Add padding at the bottom layer to maintain the original number of levels
    pad_width = ((0, 0), (1, 0))  # Pad before the first vertical level
    # shear_magnitude_init = np.pad(shear_magnitude_init, pad_width, mode='constant', constant_values=0)
    # shear_direction_init = np.pad(shear_direction_init, pad_width, mode='constant', constant_values=0)
    # shear_magnitude_max = np.pad(shear_magnitude_max, pad_width, mode='constant', constant_values=0)
    # shear_direction_max = np.pad(shear_direction_max, pad_width, mode='constant', constant_values=0)

    # shear_tangeant_init = np.pad(shear_tangeant_init, pad_width, mode='constant', constant_values=0)
    # shear_normal_init = np.pad(shear_normal_init, pad_width, mode='constant', constant_values=0)
    # shear_tangeant_max_instant = np.pad(shear_tangeant_max_instant, pad_width, mode='constant', constant_values=0)
    # shear_normal_max_instant = np.pad(shear_normal_max_instant, pad_width, mode='constant', constant_values=0)

    # shear_magnitude_init_bis = np.pad(shear_magnitude_init_bis, pad_width, mode='constant', constant_values=0)
    # shear_magnitude_max_bis = np.pad(shear_magnitude_max_bis, pad_width, mode='constant', constant_values=0)
    
    profiles_init_stack = np.stack([
        # TABS_init_profile[:,4:-2],
        # QV_init_profile[:,4:-2],
        moving_average(wind_tangeant_init,3),
        moving_average(wind_normal_init, 3),
        # moving_average(shear_tangeant_init, 5),
        # moving_average(shear_normal_init, 5),
        # moving_average(shear_magnitude_init, 5),
        # shear_magnitude_init_bis, it works
    ], axis=2)

    # profiles_max_stack = np.stack([
    #     moving_average(shear_tangeant_max_instant, 5), 
    #     moving_average(shear_normal_max_instant, 5),
    #     # moving_average(shear_magnitude_max, 5),
    #     # shear_magnitude_max_bis, it works
    # ], axis=2)

    # Concatenate initial and max profiles
    # X = np.concatenate([profiles_init_stack, profiles_max_stack], axis=2)
    X = profiles_init_stack
    num_samples, num_levels, num_features = X.shape

    # Reshape to 2D array for scaling
    X_reshaped = X.reshape(-1, num_features)

    # Initialize or use provided scaler
    if scaler is None:
        scaler_profiles = StandardScaler()
        X_scaled_reshaped = scaler_profiles.fit_transform(X_reshaped)
    else:
        scaler_profiles = scaler
        X_scaled_reshaped = scaler_profiles.transform(X_reshaped)

    X_scaled = X_scaled_reshaped.reshape(num_samples, num_levels, num_features)
    return X, X_scaled, scaler_profiles
