#!/usr/bin/env python

import pandas as pd
import numpy as np
import sys, os
import json
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)
from libraries.functions import *



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

    ################################### READ DATA AND COMPUTE VELOCITY AT SURFACE ###################################
    ts = 0
    all_data = pd.read_parquet(f"{gz_folder}{m}/fields/full.{ts}.gzip")

    # #compute radius for each level 
    # r = np.zeros((len(all_data),1))
    # rad = outer_radius - np.sqrt(all_data['Points:2']**2 + all_data['Points:1']**2 + all_data['Points:0']**2)

    # print(rad/1.e3)

    # # Assume outer_radius is known (e.g., 6371e3 for Earth)
    # tolerance = 1e3  # 1 km window
    # mask = (r > outer_radius - tolerance) & (r < outer_radius + tolerance)

    # # Apply mask to positions and velocities
    # x_surf = x[mask]
    # y_surf = y[mask]
    # z_surf = z[mask]
    # u_surf = u[mask]
    # v_surf = v[mask]
    # w_surf = w[mask]

        





if __name__ == "__main__":
    main()