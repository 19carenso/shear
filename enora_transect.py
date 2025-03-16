import numpy as np
from shapely.geometry import Point, Polygon
import cartopy.crs as ccrs

def add_transects_with_aligned_boxes(ax, transects, width, color='red', linewidth=2, linestyle='-', label=None):
    """
    Draw boxes aligned with transects on the map.

    Parameters:
    - ax: The axis object to draw the transects on.
    - transects: A list of transects, where each transect is defined by its start and end points (lat, lon).
    - width: The half-width of the box around the transects in degrees.
    - color, linewidth, linestyle, label: Customization options for the transect boxes.
    """
    out_per_transect = []
    for start, end in transects:
        # Direction from start to end
        direction = np.array((end[0] - start[0], end[1] - start[1]))
        
        # Perpendicular direction to the transect
        perpendicular = np.array((-direction[1], direction[0]))
        perpendicular_direction = perpendicular / np.linalg.norm(perpendicular)
        
        # Calculate the offset for the box width
        offset = perpendicular_direction * width
        
        # Calculate box corners
        bottom_left = np.array(start) - offset
        bottom_right = np.array(end) - offset
        top_left = np.array(start) + offset
        top_right = np.array(end) + offset
        
        # Define box corners and close the loop
        box_corners = [bottom_left, bottom_right, top_right, top_left, bottom_left]
        
        # Separate lat and lon for plotting
        box_lats, box_lons = zip(*box_corners)
        
        # Plot the box
        ax.plot(box_lons, box_lats, color=color, linewidth=linewidth, linestyle=linestyle,
                transform=ccrs.PlateCarree(), label=label)
        
    out_per_transect.append((box_corners, box_lats, box_lons))  
    return box_corners[:-1]

def make_mask_box(start, end, width):
    ## make a mask on lon lat grid, as a box given start (lat,lon) and end points
    # Direction from start to end
    direction = np.array((end[0] - start[0], end[1] - start[1]))

    # Perpendicular direction to the transect
    perpendicular = np.array((-direction[1], direction[0]))
    perpendicular_direction = perpendicular / np.linalg.norm(perpendicular)

    # Calculate the offset for the box width
    offset = perpendicular_direction * width

    # Calculate box corners
    bottom_left = np.array(start) - offset
    bottom_right = np.array(end) - offset
    top_left = np.array(start) + offset
    top_right = np.array(end) + offset

    # Define box corners and close the loop
    box_corners = [bottom_left, bottom_right, top_right, top_left, bottom_left]

    # Separate lat and lon for plotting
    box_lats, box_lons = zip(*box_corners)
    return box_corners

################
## Parameters ##
################

box_width = 0.1
n_bins = 10

data = ? 

transects = [((lat_start, lon_start), (lat_end, lon_end))] 

## For plottings the big transect
add_transects_with_aligned_boxes(ax, transects, width=box_width, color='r', linewidth=1, linestyle='--')  # Draw the transects on the map

data_transect = np.full(fill_value = float(0), shape=(z_dim, n_bins)) #for exemple a transect with vertical axis. so ndim =2

# for loop not necessary, it's for applying code to transects list with multiple values
for transect in transects: 
    start_transect, end_transect = transect[0], transect[1]

    bin_sides = np.linspace(start_transect, end_transect, n_bins+1) 
    
    for i_bin, bin_box in enumerate(zip(bin_sides[:-1], bin_sides[1:])): 
        bin_box_corners = make_mask_box(bin_box[0], bin_box[1], box_width)
        bin_box_lats, bin_box_lons = zip(*bin_box_corners)

        # if you want to plot the box in which things are computed along the transect
        ax.plot(bin_box_lons, bin_box_lats, color='r', linewidth=1, linestyle='--',transform=ccrs.PlateCarree(), label=bin_box_corners[0][0])

        # Create a polygon from the corners to build mask 
        polygon = Polygon(bin_box_corners)
        lat_grid, lon_grid = np.meshgrid(data['lat'].values, data['lon'].values, indexing='ij')
        mask = np.zeros(lat_grid.shape, dtype=bool)
        for i_lat in range(lat_grid.shape[0]):
            for j_lat in range(lat_grid.shape[1]):
                point = Point(lat_grid[i_lat, j_lat], lon_grid[i_lat, j_lat])
                if polygon.contains(point):
                    mask[i_lat, j_lat] = True

        if not np.all(~mask): 
            data_box = data.where(mask).mean(dim=['lat', 'lon'])
            data_transect[:, i_bin] = data_box

