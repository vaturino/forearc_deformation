#! /usr/bin/env python

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import os

min_lat = -60
max_lat = 20
min_lon = 220
max_lon = 320
dx = 0.2  # grid spacing (degree)   # keep same spacing with slab geometry

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
                    
                    # Check for duplicates or invalid data (latitude and longitude ranges)
                    if not (-180 <= lon <= 180) or not (-90 <= lat <= 90):
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

# Path to your .dig file
file_path = '/home/vturino/PhD/projects/forearc_deformation/plates/pb2002_plates.dig'  # Update with actual path

# Read the plate boundary data from the .dig file
plate_boundaries = read_plate_boundary_file(file_path)

# Define the desired ranges for longitude and latitude
new_lon = np.linspace(min_lon, max_lon, int((max_lon - min_lon) / dx) + 1, endpoint=True, dtype=np.float32)  # Longitude range
new_lat = np.linspace(min_lat, max_lat, int((max_lat - min_lat) / dx) + 1, endpoint=True, dtype=np.float32)  # Latitude range

# Create the map
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Set the extent to the desired longitude and latitude range
ax.set_extent([250, 320, -60, 20], crs=ccrs.PlateCarree())  # Zoom into the region you specified

# Add gridlines and coastlines
ax.gridlines(draw_labels=True)
ax.coastlines()

# Plot only NZ and SA plate boundaries
for region, coords in plate_boundaries.items():
    if region in regions_to_plot:  # Only plot NZ and SA
        lon, lat = zip(*coords)
        ax.plot(lon, lat, label=f'{region} Plate Boundary', color='blue', transform=ccrs.PlateCarree())

# Add title and legend
plt.title('Plate Boundaries: NZ and SA')
plt.legend()

# Show the plot
plt.show()