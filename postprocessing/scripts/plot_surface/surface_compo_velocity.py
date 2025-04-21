#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys, os
import json
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.tri import Triangulation
import seaborn as sns

from pathlib import Path
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from libraries.functions import *
from matplotlib.colors import ListedColormap


def main():

    ################################### READ IN INITIAL FILES ###################################
    parser = argparse.ArgumentParser(description= 'Script that gets n models and the time at the end of computation and gives the temperature and viscosity plots for all time steps')
    parser.add_argument('json_file', help='json file with models and end time to plot T and eta field')
    args = parser.parse_args()


    gz_folder = '/home/vturino/Vale_nas/forearc_deformation/gz_outputs/'  
    models_loc = '/home/vturino/Vale_nas/forearc_deformation/raw_outputs/'  
    json_loc = '/home/vturino/PhD/projects/forearc_deformation/postprocessing/json_files/'
    plt_folder = '/home/vturino/PhD/projects/forearc_deformation/postprocessing/plots/single_models/'
    if not os.path.exists(plt_folder):
        os.makedirs(plt_folder)

    with open(f"{json_loc}{args.json_file}") as json_file:
            configs = json.load(json_file)

    m = configs['model_name']
    slines = configs['head_lines']

    ################################### GLOBAL VARIABLES ###################################
    outer_radius = 6371.0e3
    inner_radius = 5271.0e3

    # Color palette for compositions
    palette_by_composition = {
        'background': 'gainsboro',     
        'op': 'yellowgreen',      
        'crust': 'olivedrab',         
        'plbd': 'darkgreen'        
    }

    # Fixed composition field names
    compo_field_mapping = {
        0: 'Background',
        1: 'Plate Boundaries',
        2: 'Oceanic Crust',
        3: 'Overriding Plate'
    }
        

    ################################### GET TIME ARRAY ###################################
    time_array = np.zeros((len(os.listdir(f"{gz_folder}{m}/fields")),2))
    stat = pd.read_csv(f"{models_loc}{m}/statistics",skiprows=slines,sep='\s+',header=None)
    time_array = grab_dimTime_fields(f"{gz_folder}{m}/fields", stat, time_array, slines-1)
    plot_loc_mod = f"{plt_folder}/{m}"

    if not os.path.exists(plot_loc_mod):
        os.mkdir(plot_loc_mod)
    plot_loc = f"{plot_loc_mod}/Surface_velocities/"

    if not os.path.exists(plot_loc):
        os.mkdir(plot_loc)
    else:
        # Count the files in the fields_loc directory
        file_count = len(os.listdir(plot_loc))

    ################################### READ DATA AND PLOT COMPOSITION AT SURFACE ###################################
    for ts in range(len(time_array)):
        all_data = pd.read_parquet(f"{gz_folder}{m}/fields/full.{ts}.gzip")

        # Calculate the radial distance for the entire dataset
        r_all = np.sqrt(all_data['Points:0']**2 + all_data['Points:1']**2 + all_data['Points:2']**2)

        # Find the surface points (within the tolerance)
        tolerance = 0.0  # 0 tolerance for surface points
        surf_idx = np.where(np.abs(r_all - outer_radius) <= tolerance)[0]

        # Filter surface data
        surf_data = all_data.iloc[surf_idx]

        # Get x, y, z, and compositions for the surface points
        x = surf_data['Points:0'].values
        y = surf_data['Points:1'].values
        z = surf_data['Points:2'].values

        # Convert to spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        lat = np.degrees(np.arcsin(z / r))
        lon = np.degrees(np.arctan2(y, x))
        lon = (lon + 360) % 360

        # Normalize compositions to 0 or 1
        crust = (surf_data['crust'].values > 0.5).astype(int)
        op = (surf_data['op'].values > 0.5).astype(int)
        plbd = (surf_data['plbd'].values > 0.5).astype(int)

        # Surface velocities (convert to cm/yr)
        vx = surf_data['velocity:0'].values * 100  # cm/yr
        vy = surf_data['velocity:1'].values * 100
        vz = surf_data['velocity:2'].values * 100

        # Radial velocity calculation (dot product for velocity along radial direction)
        vr = (x * vx + y * vy + z * vz) / r

        # Latitude and longitude velocity components (using the correct r for surface points)
        lon = np.radians(lon)
        lat = np.radians(lat)
        # v_lon = (vx * np.sin(lon) - vy * np.cos(lon)) / (r * np.cos(lat))
        # v_lat = (vx * np.cos(lat) + vy * np.sin(lat)) / r

        # Combine compositions into one array
        compo = crust + 2*op + 3*plbd

        # Convert to spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)
        lat = np.degrees(np.arcsin(z / r))
        lon = np.degrees(np.arctan2(y, x))
        lon = (lon + 360) % 360

        # Triangulation and plotting
        triang = Triangulation(lon, lat)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = [palette_by_composition[key] for key in sorted(palette_by_composition.keys())]
        custom_cmap = ListedColormap(colors)

        tpc = ax.tripcolor(triang, compo, cmap=custom_cmap, shading='gouraud', vmin=0, vmax=3)
        cbar = plt.colorbar(tpc, label='Composition')
        cbar.set_ticks(list(compo_field_mapping.keys()))
        cbar.set_ticklabels([compo_field_mapping[i] for i in compo_field_mapping.keys()])

        # Number of points to skip (e.g., every 10th point)
        skip = 2500

        # Subsample the data
        lon_subsampled = lon[::skip]
        lat_subsampled = lat[::skip]
        vx_subsampled = vx[::skip]
        vy_subsampled = vy[::skip]

        # Plot the velocity vectors (subsampled)
        q = ax.quiver(lon_subsampled, lat_subsampled, vx_subsampled, vy_subsampled, scale=0.25, scale_units='xy', 
            angles='xy', color='black', headwidth=2, width = 0.003, label='Velocity')

        ref_velocity = 2  # cm/yr
        ref_x = lon.min() + 1  # Adjust position near bottom-left corner
        ref_y = lat.min() + 1
        ax.quiver(ref_x, ref_y, ref_velocity, 0, scale=0.25, scale_units='xy', 
                angles='xy', color='black', headwidth=2, width = 0.003, label='2 cm/yr')
        ax.text(ref_x + 1, ref_y+2, '2 cm/yr', fontsize=8, verticalalignment='center')


        # Axes and labels
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Surface composition and velocity at t = {round(time_array[ts,1]/1.e6, 2)} Myr")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(lon.min(), lon.max())
        ax.set_ylim(lat.min(), lat.max())
        plt.tight_layout()
        plt.savefig(f"{plot_loc}/surface_composition_and_velocity_{int(ts)}.png", dpi=300)
        plt.close('all')


if __name__ == "__main__":
    main()
