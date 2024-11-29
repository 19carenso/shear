from sklearn.preprocessing import StandardScaler
import numpy as np

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