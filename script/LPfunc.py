# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:51:37 2022

@author: richa
"""

import sys, os, glob
import pandas as pd
from tqdm import tqdm
import pickle as pkl
import numpy as np
import scipy
import librosa
import librosa.display
import re
import soundfile
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic
import fnmatch



def power_frames(signal, fs, frame, step):
    """
    Compute power of each time frame. The original signal is firstly divided
    into several time frames 'frame' with the hop length/step size 'step', and 
    then signal power is computed based on each time frame of the signal.

    Parameters
    ----------
    signal : 1-D array
        the original audio file
    fs : int
        sampling frequency (Hz)
    frame : int
        time length of each time frame
    step : int
       number of samples to advance for each frame

    Returns
    -------
    power_matrix: list
        a list contains all power values of all time frames

    """
    power_matrix = []
    i = 0
    while i+frame <= len(signal):
        p_frame = sum(abs(signal[i:i+frame])**2)/(frame/fs)
        power_matrix.append(p_frame)
        i += step
    return power_matrix

def find_setf(signal, fs, frame=2000, step=200):
    """
    Find time indexes that represent the onset and offset of each cough event.

    Parameters
    ----------
    signal : 1-D array
        the original audio file
    fs : int
        sampling frequency (Hz)
    frame : int
        time length of each time frame
    step : int
        number of samples to advance for each frame
    thres : float
        The threshold used for picking cough events and filter out the silence frames.
        
    Returns
    -------
    onsetf : 1-D array
        Time indexes (unit:second) of onsets of cough events.
    offsetf : 1-D array
        Time indexes (unit:second) of offsets of cough events.

    """
    power_matrix = power_frames(signal, fs, frame=frame, step=step)
    thres = 1
    indices = [idx for idx,val in enumerate(power_matrix) if val > thres]
    i = 0
    j = 1
    onsetf = []
    offsetf = []
    while i < len(indices)-1 and j <len(indices)-1:
        if (indices[j] - indices[i]) == (j - i):
            j += 1
        else:
            onsetf.append(indices[i])
            offsetf.append(indices[j-1])
            i = j
            j += 1
    onsetf.append(indices[i])
    offsetf.append(indices[j])
    onset = (np.array(onsetf)*step+0.75*frame)/fs #start point of the frame
    offset = (np.array(offsetf)*step+1*frame)/fs #end point of the frame
    return onset, offset

def find_voice(signal, fs):
    """
    Detect all single cough events

    Parameters
    ----------
    signal : 1-D array
        the original audio file
    fs : int
        sampling frequency (Hz)
    frame : int
        time length of each time frame
    step : int
        number of samples to advance for each frame
    thres : float
        The threshold used for picking cough events and filter out the silence frames.
        
    Returns
    -------
    cough_event : List
        Contains all single cough events

    """
    onset, offset = find_setf(signal, fs)
    cough_event = []
    for i in range(len(onset)):
        # cough_length = int(offset[i]*fs) - int(onset[i]*fs)
        # if cough_length > 0.02*fs:
            cough_event.append(signal[int(onset[i]*fs):int(offset[i]*fs)])
    
    return cough_event

def remove_silence(signal, fs):
    """
    Return a signal without silent parts

    Parameters
    ----------
    signal : TYPE
        DESCRIPTION.
    fs : TYPE
        DESCRIPTION.
    frame : TYPE, optional
        DESCRIPTION. The default is 1024.
    step : TYPE, optional
        DESCRIPTION. The default is 100.
    thres : TYPE, optional
        DESCRIPTION. The default is 8.

    Returns
    -------
    voiced_signal : TYPE
        DESCRIPTION.

    """
    segments = find_voice(signal, fs)
    voiced_signal = np.concatenate(segments)
    
    return voiced_signal


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
    

def get_voice_fea(sound, normalize='no', lpc_order=18):
     
    # use pitch tracking to get voiced frame indexes
    signal = basic.SignalObj(sound)
    pitch = pYAAPT.yaapt(signal,**{'f0_min' : 30.0, 'frame_length' : 15.0, 'frame_space' : 5.0})
    
    # normalize amplitude between -1 and 1
    if normalize == 'yes':
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


def get_residual_fea(sound,normalize='no',lpc_order=18):
  
    signal = basic.SignalObj(sound)
    pitch = pYAAPT.yaapt(signal,**{'f0_min' : 30.0, 'frame_length' : 15.0, 'frame_space' : 5.0})
    voi_pos = pitch.frames_pos[np.where(pitch.samp_values!=0)]
    frame_size = pitch.frame_size
    
    # normalize amplitude between -1 and 1
    if normalize == 'yes':
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


def get_unvoice_fea(sound,normalize='no',lpc_order=18):
  
    signal = basic.SignalObj(sound)
    pitch = pYAAPT.yaapt(signal,**{'f0_min' : 30.0, 'frame_length' : 15.0, 'frame_space' : 5.0})
    unv_pos = pitch.frames_pos[np.where(pitch.samp_values==0)]
    frame_size = pitch.frame_size
    
    # normalize amplitude between -1 and 1
    if normalize == 'yes':
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


""" 
Remove silence from original recordings and create new recordings.
"""

def load_one_ad(ad_name,fs=16000):
    
    ad, _ = librosa.load(ad_name,sr=fs)
    ad = ad/np.max(abs(ad)) #amplitude normalize between -1 and 1
    return ad


def save_one_ad(folder_path,ad_name,ad,fs=16000):
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    soundfile.write(os.path.join(folder_path,ad_name),ad,fs)
    
    return 0


def process_ad_folder(folder_path_og,folder_path_new,ad_type:str,fs=16000):
    
    for file in tqdm(sorted(glob.glob(os.path.join(folder_path_og, '*.%s'%(ad_type))))):
        
        filename = os.path.basename(file).split('/')[-1] #get filename from path
        
        """ load one audio file from folder """
        data = load_one_ad(ad_name=os.path.join(folder_path_og,filename),fs=fs)
        
        if len(data) >= fs:
            """ remove silence frames only if recordings have length > 1s"""
            new_data = remove_silence(signal=data,fs=fs)
        
        elif len(data) < fs:
            new_data = data
            print(file) #see which files are shorter than 1s; could be biased files

        """ save edited audio file in target folder """
        if filename.endswith('.flac'):
            filename = filename[:-5]
            
        new_name = '%s.wav'%(filename)
        save_one_ad(folder_path=folder_path_new,
                    ad_name=new_name,
                    ad=new_data,
                    fs=fs)
    
    print('--------')
    print('Audio files saved in ' + folder_path_new)
        
    return 0 
        
    
"""
Extract LP features from new recordings.
"""
def extract_lp(ad,**kwargs):
    
    """ LP coefficients and pitch descriptors of voiced frames """
    lp_1 = get_voice_fea(sound=ad,
                         normalize=kwargs['normalize'],
                         lpc_order=int(kwargs['lp_order']))
    
    """ LP residual features of voiced frames """
    lp_2 = get_residual_fea(sound=ad,
                            normalize=kwargs['normalize'],
                            lpc_order=int(kwargs['lp_order']))
    
    """ LP residual features of unvoiced frames """
    lp_3 = get_unvoice_fea(sound=ad,
                           normalize=kwargs['normalize'],
                           lpc_order=int(kwargs['lp_order']))
    
    lp_fea = np.concatenate((lp_1,lp_2,lp_3),axis=1) # aggregate voiced and unvoiced features
    
    assert np.nan not in lp_fea, "NaN in LP feature value"
    
    return lp_fea


def extract_lp_in_group(ad_path,ad_type:str,fs=16000, **kwargs):
    
    if not os.path.exists(ad_path):
        raise Exception('folder path does not exist!')
    
    num_ad = len(fnmatch.filter(os.listdir(ad_path), '*.%s'%(ad_type)))
    
    if num_ad == 0:
        raise Exception('No audio file in the folder!')
        
    lp_group = np.empty((num_ad,36))*np.nan
    
    """ extract LP features from recordings in the folder """
    for i, file in tqdm(enumerate(sorted(glob.glob(os.path.join(ad_path, '*.%s'%(ad_type)))))):
        
        lp_group[i,:] = extract_lp(file,**kwargs) 
    
    assert np.nan not in lp_group, "NaN in group LP features"
    
    return lp_group


def save_data(data_path,data_name,data):
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    # Store data (serialize)
    with open(os.path.join(data_path,'%s.pkl'%(data_name)), 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    print("--------")
    print('Feature saved in '+data_path)
    
    return 0
