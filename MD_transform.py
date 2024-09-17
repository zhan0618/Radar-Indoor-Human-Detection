import torch
import numpy as np

def remove_static(data, zero_doppler_channels=3):
    """Remove middle doppler channels"""
    data = np.concatenate((data[64-zero_doppler_channels,:],data[-64+zero_doppler_channels:,:]),axis=0)
    return data

def remove_outer(data,outer_channels=24):
    """Remove outer channels"""
    data = data[outer_channels:128-outer_channels,:]
    return data
def threshold(data,threshold_value=15):
    """Threshold to remove significant noise"""
    #data_db =  (data > 25) * data
    data[data<=threshold_value] = threshold_value
    return data

def reverse(data):
    """Reverse data"""
    data = np.flip(data,axis=0).copy()
    return data