import os
import re
import csv
import math
import argparse

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset

import h5py
import numpy as np

from tqdm import trange
from tqdm import tqdm

'''
Process each bin file to hdf5
'''
def create_folder(folder):
    h5_folder = os.path.join(folder, 'rm_bg_md_h5_files')
    if not os.path.exists(h5_folder):
        os.makedirs(h5_folder)

def file_exist(folder,file):
    file_path = os.path.join(folder,'rm_bg_md_h5_files',file.replace('bin','h5'))
    if os.path.isfile(file_path):
        return True
    else:
        return False

def checkMovingTarget(file):
        if len(re.split(r'_',file)) > 3:
            if re.split(r'_',file)[3][0] in list('abcde'):
                return True
            else:
                return False
        else:
            return False

def get_bg_data(root):
    bg = {}
    for file in os.listdir(os.path.join(folder,'bin_files')):
        if not re.match(r'.*\.bin',file):
            continue
        gt = re.split(r'_',file)[1][0]
        env = re.split(r'_',file)[0]
        if gt == "B":
            bin_file = os.path.join(folder,'bin_files', file)
            data = np.fromfile(bin_file, dtype=np.short, count=-1)
            nFrame = int(math.floor(data.size / (setting['ADC_SAMPLES'] * setting['NCHIRPS'] * setting['NVX'] * 2)))
            data = data[0:(nFrame*setting['ADC_SAMPLES'] * setting['NCHIRPS'] * setting['NVX'] * 2)]
            data = data.reshape(-1,4)
            adc_data = np.zeros((data.shape[0],2),dtype=complex)
            adc_data[:,0] = data[:,0] + data[:,2]*1j
            adc_data[:,1] = data[:,1] + data[:,3]*1j
            pre_adc_data = adc_data.reshape(nFrame,setting['NCHIRPS'],setting['NVX'],setting['ADC_SAMPLES'])
            pre_adc_data = pre_adc_data.transpose((0,2,1,3)) # frame,vx,chirps,adc_samples
            bg[env] = pre_adc_data[:800,:,:,:]
    return bg


def get_bin_list(root):
    """
    return 
    ['1.bin', '2.bin']
    """
    file_list = os.listdir(os.path.join(root,'bin_files'))
    bin_list = []
    for f in file_list:
        if not re.match(r'.*\.bin',f):
            continue
        if checkMovingTarget(f):
            bin_list.append(f)
    return(bin_list)


# def _rdmap():
def bin2h5(folder,file,bg_data):
    bin_file = os.path.join(folder,'bin_files', file)
    data = np.fromfile(bin_file, dtype=np.short, count=-1)
    nFrame = int(math.floor(data.size / (setting['ADC_SAMPLES'] * setting['NCHIRPS'] * setting['NVX'] * 2)))
    data = data[0:(nFrame*setting['ADC_SAMPLES'] * setting['NCHIRPS'] * setting['NVX'] * 2)]
    data = data.reshape(-1,4)
    adc_data = np.zeros((data.shape[0],2),dtype=complex)
    adc_data[:,0] = data[:,0] + data[:,2]*1j
    adc_data[:,1] = data[:,1] + data[:,3]*1j
    pre_adc_data = adc_data.reshape(nFrame,setting['NCHIRPS'],setting['NVX'],setting['ADC_SAMPLES'])
    pre_adc_data = pre_adc_data.transpose((0,2,1,3)) # frame,vx,chirps,adc_samples

    # remove the DC component
    pre_adc_data = pre_adc_data - bg_data[re.split(r'_',file)[0]][:800,:,:,:]

    pre_adc_data = np.mean(pre_adc_data,axis=1)
    range_fft = np.fft.fft(np.multiply(pre_adc_data,fast_time_window),axis=2)
    rdmap = np.fft.fftshift(np.fft.fft(np.multiply(np.transpose(range_fft,(0,2,1)),slow_time_window), axis=2),axes=2)

    rdmap = np.abs(rdmap)
    rdmap = 20*np.log10(rdmap)

    # import matplotlib.pyplot as plt
    # plt.imshow(rdmap[0,:,:])
    # plt.show()



    stride = setting['stride'] 
    timesteps=setting['timesteps'] 
    noSamples = math.floor((rdmap.shape[0]-timesteps)/stride)+1
    md = np.empty((noSamples,setting['NCHIRPS'],timesteps))
    for i in range(noSamples):
        tmp = rdmap[i*stride:i*stride+timesteps,:,:]
        md[i,:,:] = tmp.mean(axis=1).transpose((1,0))

    #rdmap[rdmap<=15]=0
    #rdmap = rdmap[:,:,:,setting['zero_doppler_index'][0]:setting['zero_doppler_index'][1]]
    
    h5_name = os.path.join(folder,'rm_bg_md_h5_files',file.replace('bin','h5'))
    # Open an HDF5 file for writing
    with h5py.File(h5_name, "w") as f:
        # Create a dataset in the file to store the tensor
        dset = f.create_dataset("tensor", data=md)
    del rdmap,pre_adc_data, data, adc_data, md



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='bin2hdf5')
    parser.add_argument('--folder', type=str, default=False,help='folder')
    parser.add_argument('--bin',type=str,default=False,help='bin file name')

    folder = parser.parse_args().folder
    bin_file = parser.parse_args().bin

    setting = {'ADC_SAMPLES': 256,
               'NCHIRPS': 128,
               'NVX': 8,
               'stride':25,
               'timesteps':45}
    
    fast_time_window = np.blackman(setting['ADC_SAMPLES'])
    fast_time_window = fast_time_window/np.sum(fast_time_window)
    slow_time_window = np.blackman(setting['NCHIRPS'])
    slow_time_window = slow_time_window/np.sum(slow_time_window)
    
    if folder:
        file_list = get_bin_list(folder)
        bg_data = get_bg_data(folder)
        create_folder(folder)
        for file in tqdm(file_list):
            if not file_exist(folder,file):
                bin2h5(folder,file,bg_data)
    if bin_file:
        data = np.fromfile(bin_file, dtype=np.short, count=-1)
        nFrame = int(math.floor(data.size / (setting['ADC_SAMPLES'] * setting['NCHIRPS'] * setting['NVX'] * 2)))
        data = data[0:(nFrame*setting['ADC_SAMPLES'] * setting['NCHIRPS'] * setting['NVX'] * 2)]
        data = data.reshape(-1,4)
        adc_data = np.zeros((data.shape[0],2),dtype=complex)
        adc_data[:,0] = data[:,0] + data[:,2]*1j
        adc_data[:,1] = data[:,1] + data[:,3]*1j
        pre_adc_data = adc_data.reshape(nFrame,setting['NCHIRPS'],setting['NVX'],setting['ADC_SAMPLES'])
        pre_adc_data = pre_adc_data.transpose((0,2,1,3)) # frame,vx,chirps,adc_samples
        pre_adc_data = np.mean(pre_adc_data,axis=1)

        range_fft = np.fft.fft(np.multiply(pre_adc_data,fast_time_window),axis=2)
        rdmap = np.fft.fftshift(np.fft.fft(np.multiply(np.transpose(range_fft,(0,2,1)),slow_time_window), axis=2),axes=2)

        rdmap = np.abs(rdmap)
        rdmap = 20*np.log10(rdmap)

        stride = setting['stride'] 
        timesteps=setting['timesteps'] 
        noSamples = math.floor((rdmap.shape[0]-timesteps)/stride)+1
        md = np.empty((noSamples,setting['NCHIRPS'],timesteps))
        for i in range(noSamples):
            tmp = rdmap[i*stride:i*stride+timesteps,:,:]
            md[i,:,:] = tmp.mean(axis=1).transpose((1,0))
        with h5py.File("tensor.h5", "w") as f:
            # Create a dataset in the file to store the tensor
            dset = f.create_dataset("tensor", data=md)

