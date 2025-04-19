#! /usr/bin/python3
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import math
import matplotlib.pyplot as plt
import sys, os, subprocess
from scipy.signal import savgol_filter

def load_data (loc_data: str, init: str, compo: list, tr = 1e-2):
    indata = pd.read_parquet(loc_data+init)
    indata = indata[(indata[compo] > tr).any(axis=1)]
    return indata

def get_trench_position_from_op(p, threshold = 2.7e7):
    if {"opc"}.issubset(p.columns):
        tr =  p.loc[(p['Points:0']< threshold) & (p["opc"] > 0.3) & (p["Points:1"] >=  p["Points:1"].max() - 2.e3),'Points:0'].min()
    else:
        tr =  p.loc[(p['Points:0']< threshold) & (p['op'] > 0.3) & (p["Points:1"] >=  p["Points:1"].max() - 2.e3),'Points:0'].min()
    return tr

def collect_particles (trench:float, min_d: float, max_d: float, init, fin):
    data = init[(init["Points:0"] < trench - min_d) & (init["Points:0"] > trench - max_d) & (init["opc"] == 0)]
    data = data[data["id"].isin(fin["id"])]
    return data


def get_trench_position_from_op(p, threshold = 2.7e7):
    if {"opc"}.issubset(p.columns):
        tr =  p.loc[(p['Points:0']< threshold) & (p["opc"] > 0.3) & (p["Points:1"] >=  p["Points:1"].max() - 2.e3),'Points:0'].min()
    else:
        tr =  p.loc[(p['Points:0']< threshold) & (p['op'] > 0.3) & (p["Points:1"] >=  p["Points:1"].max() - 2.e3),'Points:0'].min()
    return tr