## imports
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 
import xarray as xr 



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