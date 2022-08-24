# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:51:37 2022

@author: richa
"""

import sys, os, glob
import pandas as pd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.signal import butter, filtfilt, hilbert
from am_analysis_box import am_analysis as ama
import librosa
import librosa.display
import re
import soundfile
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
from matplotlib import pyplot as plt



def entropy(x):
    """
    x is assumed to be an (nsignals, nsamples) array containing integers between
    0 and n_unique_vals
    """
    _, inx = np.unique(x,return_inverse = True)
    counts = np.bincount(inx)

    return scipy.stats.entropy(counts)


def get_lpc(data,order=18):
    d = data
    a = librosa.lpc(d, order)
    b = np.hstack([[0], -1 * a[1:]])
    d_hat = scipy.signal.lfilter(b, [1], d)
    err = d-d_hat
    
    return a[1:], err
    

def get_voice_fea(sound, normalize=False, lpc_order=18):
     
    # use pitch tracking to get voiced frame indexes
    signal = basic.SignalObj(sound)
    pitch = pYAAPT.yaapt(signal,**{'f0_min' : 30.0, 'frame_length' : 15.0, 'frame_space' : 5.0})
    
    # normalize amplitude between -1 and 1
    if normalize:
        data = signal.data/np.max(signal.data)
    else:
        data = signal.data
    
    voice_fea = np.zeros((1,lpc_order+2))*np.nan
    
    # get lpc coefficients
    coef, _ = get_lpc(data,order=lpc_order)
    voice_fea[0,:-2] = coef
    
    # get pitch mean and std
    voice_fea[0,-2] = np.mean(pitch.samp_interp)
    voice_fea[0,-1] = np.std(pitch.samp_interp)
    
    return voice_fea


def get_residual_fea(sound,normalize=False,lpc_order=18):
  
    signal = basic.SignalObj(sound)
    pitch = pYAAPT.yaapt(signal,**{'f0_min' : 30.0, 'frame_length' : 15.0, 'frame_space' : 5.0})
    voi_pos = pitch.frames_pos[np.where(pitch.samp_values!=0)]
    frame_size = pitch.frame_size
    
    # normalize amplitude between -1 and 1
    if normalize:
        data = signal.data/np.max(signal.data)
    else:
        data = signal.data
        
    # get lp residuals
    _, err = get_lpc(data,order=lpc_order)
    
    # get residual descriptors for each unvoiced frame
    m = []
    std = []
    kur = []
    skew = []
    # enp = []
    for pos in voi_pos:
        residual = err[int(pos-frame_size/2):int(pos+frame_size/2)]

        m.append(np.mean(residual))
        std.append(np.std(residual))
        kur.append(scipy.stats.kurtosis(residual))
        skew.append(scipy.stats.skew(residual))
        # enp.append(entropy(residual))
        
    res_fea = np.zeros((1,8))*np.nan

    res_fea[:,0] = np.mean(m)
    res_fea[:,1] = np.mean(std)
    res_fea[:,2] = np.mean(kur)
    res_fea[:,3] = np.mean(skew)
    res_fea[:,4] = np.std(m)
    res_fea[:,5] = np.std(std)
    res_fea[:,6] = np.std(kur)
    res_fea[:,7] = np.std(skew)
    
    return res_fea


def get_unvoice_fea(sound,normalize=False,lpc_order=18):
  
    signal = basic.SignalObj(sound)
    pitch = pYAAPT.yaapt(signal,**{'f0_min' : 30.0, 'frame_length' : 15.0, 'frame_space' : 5.0})
    unv_pos = pitch.frames_pos[np.where(pitch.samp_values==0)]
    frame_size = pitch.frame_size
    
    # normalize amplitude between -1 and 1
    if normalize:
        data = signal.data/np.max(signal.data)
    else:
        data = signal.data
        
    # get lp residuals
    _, err = get_lpc(data,order=lpc_order)
    
    # get residual descriptors for each unvoiced frame
    m = []
    std = []
    kur = []
    skew = []
    # enp = []
    for pos in unv_pos:
        residual = err[int(pos-frame_size/2):int(pos+frame_size/2)]
        m.append(np.mean(residual))
        std.append(np.std(residual))
        kur.append(scipy.stats.kurtosis(residual))
        skew.append(scipy.stats.skew(residual))
        # enp.append(entropy(residual))
        
    unvoice_fea = np.zeros((1,8))*np.nan
    
    unvoice_fea[:,0] = np.mean(m)
    unvoice_fea[:,1] = np.mean(std)
    unvoice_fea[:,2] = np.mean(kur)
    unvoice_fea[:,3] = np.mean(skew)
    unvoice_fea[:,4] = np.std(m)
    unvoice_fea[:,5] = np.std(std)
    unvoice_fea[:,6] = np.std(kur)
    unvoice_fea[:,7] = np.std(skew)
    
    return unvoice_fea
