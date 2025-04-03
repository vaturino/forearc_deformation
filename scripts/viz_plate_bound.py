#! /usr/bin/env python

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os
import pandas as pd
from scipy.io import savemat


# Function to convert longitude from -180 to 180 to 0 to 360
def convert_longitude(lon):
    if lon < 0:
        return lon + 360  # Convert negative longitude to 0-360 range
    return lon  # Keep positive longitude as it is

# Function to read the plate boundary coordinates from a .dig file
def read_plate_boundary_file(file_path):
    plate_boundaries = {}
    current_region = None
    current_coords = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if '*** end of line segment ***' in line:
                continue  # Skip lines with the end of line segment marker
            
            if line.isalpha():  # Region identifier (e.g., NZ, SA)
                if current_region:  # Save previous region's coordinates if it exists
                    plate_boundaries[current_region] = current_coords
                current_region = line
                current_coords = []  # Reset for the new region
            
            elif line:  # Coordinates in 'longitude,latitude' format
                try:
                    lon, lat = map(float, line.split(','))
                    
                    # Convert the longitude from -180 to 180 to 0 to 360
                    lon = convert_longitude(lon)
                    
                    # Check for duplicates or invalid data (latitude and longitude ranges)
                    if not (-90 <= lat <= 90):
                        print(f"Skipping invalid coordinates: {lon}, {lat}")
                        continue
                    
                    # Prevent duplicate consecutive points which can cause horizontal lines
                    if current_coords and current_coords[-1] == (lon, lat):
                        continue
                    
                    current_coords.append((lon, lat))
                except ValueError:
                    print(f"Skipping line due to invalid coordinates: {line}")
        
        if current_region:  # Save the last region's coordinates
            plate_boundaries[current_region] = current_coords
    
    return plate_boundaries


def find_closest_nz_coord(sa_lon, sa_lat, nz_coords):
    # Calculate the absolute difference in longitude and latitude
    distances = nz_coords.apply(
        lambda row: (abs(row['lon'] - sa_lon) + abs(row['lat'] - sa_lat)), axis=1
    )
    # Return the index of the closest NZ coordinate
    closest_index = distances.idxmin()
    return nz_coords.loc[closest_index]

def is_close_enough(coord1, coord2, tolerance=1e-6):
    return abs(coord1['lon'] - coord2['lon']) < tolerance and abs(coord1['lat'] - coord2['lat']) < tolerance




#############################################################################################################################################


# Define the regions to plot
regions_to_plot = ['NZ', 'SA']  # List of regions to plot

# Path to your .dig file
file_path = '/home/vturino/PhD/projects/forearc_deformation/plates/pb2002_plates.dig'  # Update with actual path

# Read the plate boundary data from the .dig file
plate_boundaries = read_plate_boundary_file(file_path)


# Define the min and max latitudes and longitudes
min_lon, max_lon = 240, 320  # Min and max longitude range
min_lat, max_lat = -60, 20   # Min and max latitude range

r_min_lon, r_max_lon = 280, 300  # Min and max longitude range for the mods
r_min_lat, r_max_lat = -30, -10   # Min and max latitude range for the mods

# Initialize arrays to store coordinates for each region
coords_NZ = []
coords_SA = []

# Extract the coordinates for NZ and SA regions
for region, coords in plate_boundaries.items():
    if region == 'NZ':
        coords_NZ = coords  # Save NZ coordinates into coords_NZ
    elif region == 'SA':
        coords_SA = coords  # Save SA coordinates into coords_SA

coords_NZ = pd.DataFrame(coords_NZ)
coords_SA = pd.DataFrame(coords_SA)

coords_NZ.columns = ['lon', 'lat']
coords_SA.columns = ['lon', 'lat']

#limit coordinates to the defined range
coords_NZ = coords_NZ[(coords_NZ['lon'] >= min_lon) & (coords_NZ['lon'] <= max_lon) & (coords_NZ['lat'] >= min_lat) & (coords_NZ['lat'] <= max_lat)]
coords_SA = coords_SA[(coords_SA['lon'] >= min_lon) & (coords_SA['lon'] <= max_lon) & (coords_SA['lat'] >= min_lat) & (coords_SA['lat'] <= max_lat)]


# Ensure the DataFrames are not empty before proceeding
if not coords_NZ.empty:
    # Continue with processing the NZ coordinates as needed
    print(f"NZ coordinates are not empty, processing...")
else:
    print(f"NZ coordinates are empty, skipping...")

# Also check for coords_SA if necessary
if not coords_SA.empty:
    print(f"SA coordinates are not empty, processing...")
else:
    print(f"SA coordinates are empty, skipping...")

# Ensure coordinates are filtered within the extent
coords_NZ = coords_NZ[(coords_NZ['lon'] >= min_lon) & (coords_NZ['lon'] <= max_lon) & 
                       (coords_NZ['lat'] >= min_lat) & (coords_NZ['lat'] <= max_lat)]
coords_SA = coords_SA[(coords_SA['lon'] >= min_lon) & (coords_SA['lon'] <= max_lon) & 
                       (coords_SA['lat'] >= min_lat) & (coords_SA['lat'] <= max_lat)]


# Copy SA coordinates so we can adjust them
coords_SA_adjusted = coords_SA.copy()

for idx, row in coords_SA_adjusted.iterrows():
    # Check if the current SA coordinates are within the defined region
    if r_min_lon <= row['lon'] <= r_max_lon and r_min_lat <= row['lat'] <= r_max_lat:
        # Find the closest matching NZ coordinates in the region
        matching_nz_coords = coords_NZ[(coords_NZ['lon'] >= r_min_lon) & 
                                       (coords_NZ['lon'] <= r_max_lon) & 
                                       (coords_NZ['lat'] >= r_min_lat) & 
                                       (coords_NZ['lat'] <= r_max_lat)]
        
        # If a matching NZ coordinate is found, replace the SA coordinates
        if not matching_nz_coords.empty:
            closest_nz_coord = find_closest_nz_coord(row['lon'], row['lat'], matching_nz_coords)
            coords_SA_adjusted.at[idx, 'lon'] = closest_nz_coord['lon']
            coords_SA_adjusted.at[idx, 'lat'] = closest_nz_coord['lat']

if not coords_SA_adjusted.iloc[0].equals(coords_SA_adjusted.iloc[-1]):
    coords_SA_adjusted = pd.concat([coords_SA_adjusted, coords_SA_adjusted.iloc[[0]]], ignore_index=True)

if not coords_NZ.iloc[0].equals(coords_NZ.iloc[-1]):
    coords_NZ = pd.concat([coords_NZ, coords_NZ.iloc[[0]]], ignore_index=True)

tolerance = 1e-6  # Define a tolerance for matching coordinates

# Create a mask that identifies coordinates that appear in both coords_SA_adjusted and coords_NZ
mask_SA_NZ = coords_SA_adjusted.apply(
    lambda row: any(is_close_enough(row, nz_row, tolerance) for _, nz_row in coords_NZ.iterrows()), axis=1)

mask_NZ_SA = coords_NZ.apply(
    lambda row: any(is_close_enough(row, sa_row, tolerance) for _, sa_row in coords_SA_adjusted.iterrows()), axis=1)

# Remove the matching points from both coords_SA_adjusted and coords_NZ
coords_SA_adjusted = coords_SA_adjusted[~mask_SA_NZ]
coords_NZ = coords_NZ[~mask_NZ_SA]




# Check points in NZ if one point is far from its neighbor >10 degrees, drop it
coords_NZ = coords_NZ.reset_index(drop=True)
coords_NZ['lon_diff'] = coords_NZ['lon'].diff().abs()
coords_NZ['lat_diff'] = coords_NZ['lat'].diff().abs()
coords_NZ = coords_NZ[(coords_NZ['lon_diff'] < 10) & (coords_NZ['lat_diff'] < 10)]
coords_NZ = coords_NZ.drop(columns=['lon_diff', 'lat_diff'])

# Check if the first and last coordinates are the same, and if so, loop them back to close the loop
if not coords_SA_adjusted.iloc[0].equals(coords_SA_adjusted.iloc[-1]):
    coords_SA_adjusted = pd.concat([coords_SA_adjusted, coords_SA_adjusted.iloc[[0]]], ignore_index=True)

if not coords_NZ.iloc[0].equals(coords_NZ.iloc[-1]):
    coords_NZ = pd.concat([coords_NZ, coords_NZ.iloc[[0]]], ignore_index=True)



#save a copy of adjusted Sa and NZ coordinates
coords_SA_adjusted.to_csv('/home/vturino/PhD/projects/forearc_deformation/plates/coords_SA_adjusted.csv', index=False)
coords_NZ.to_csv('/home/vturino/PhD/projects/forearc_deformation/plates/coords_NZ.csv', index=False)

# Convert DataFrames to dictionaries
mat_data = {
    'coords_SA_adjusted': {
        'lon': coords_SA_adjusted['lon'].to_numpy(),
        'lat': coords_SA_adjusted['lat'].to_numpy()
    },
    'coords_NZ': {
        'lon': coords_NZ['lon'].to_numpy(),
        'lat': coords_NZ['lat'].to_numpy()
    }
}

# Save to .mat files
savemat('/home/vturino/PhD/projects/forearc_deformation/plates/coords_SA_adjusted.mat', {'coords_SA_adjusted': mat_data['coords_SA_adjusted']})
savemat('/home/vturino/PhD/projects/forearc_deformation/plates/coords_NZ.mat', {'coords_NZ': mat_data['coords_NZ']})



# Create a figure and plot just the boundaries, no coastline
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(coords_NZ['lon'], coords_NZ['lat'], color='blue', linewidth=2, label='NZ Plate Boundary')
ax.plot(coords_SA_adjusted['lon'], coords_SA_adjusted['lat'], color='red', linewidth=2, label='SA Plate Boundary')
plt.show()

