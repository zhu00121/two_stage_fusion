# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 17:13:37 2021

@author: richa
"""
import numpy as np
from am_analysis_box import am_analysis as ama
from numba import jit

@jit(nopython=True)
def segsum_matrix(mat,n_mbin,n_fbin,m_stride,f_stride,m_lo=0,m_hi=20,f_lo=0,f_hi=20):
    res = np.empty((n_mbin,n_fbin)).astype(np.float32)
    
    for f in np.arange(f_lo,f_hi):
        for m in np.arange(m_lo,m_hi):
            res[f,m] = np.mean(mat[f*f_stride:(f+1)*f_stride,m*m_stride:(m+1)*m_stride])
            
    return res

def mspec_energy(signal,fs,mod_lim=20,freq_lim=8000,n_mod_bin=20,n_f_bin=20,win_size=256,fft_factor_y=8,fft_factor_x=16):
    # Compute modulation spectrogram
    mspec_data = ama.strfft_modulation_spectrogram(signal,
                                  fs=fs, 
                                  win_size=win_size, 
                                  win_shift=0.125*win_size, 
                                  fft_factor_y=fft_factor_y, 
                                  win_function_y='hamming', 
                                  fft_factor_x=fft_factor_x, 
                                  win_function_x='hamming', 
                                  channel_names=None)
    # Reshape power values of modulation spectrogram into 2-D
    MS = mspec_data['power_modulation_spectrogram']
    MS = MS[:,:,0]
    # Range of modulation frequency and conventional frequency
    mod_lim = mod_lim
    freq_lim = freq_lim
    # Convert Hz into step size for further summation
    mod_step = int((mod_lim/n_mod_bin)/mspec_data['freq_mod_delta'])
    freq_step = int((freq_lim/n_f_bin)/mspec_data['freq_delta'])
    
    # # Energies to be stored
    # mod_energy = []
    # # Total energy
    # total_energy = np.sum(MS[:,:int(mod_lim*mod_step)])
    # # Loop through two axes to compute energies
    # for freq_ax in range(n_f_bin):
    #     for mod_ax in range(n_mod_bin):
    #         # Regional energy
    #         mod_energy.append(np.sum(MS[int(freq_ax*freq_step):int((freq_ax+1)*freq_step),int(mod_ax*mod_step):int((mod_ax+1)*mod_step)]))
    
    mod = segsum_matrix(MS,n_mod_bin,n_f_bin,mod_step,freq_step)
    mod = np.reshape(mod,(400,))
    
    #Normalize by total energy
    total_energy = np.sum(mod)
    ot = mod/total_energy
    
    return ot

    
def get_mspec_descriptors(mod, mod_lim=20, freq_lim=8000, n_mod_bin=20, n_freq_bin=20):
    n_fea = 8 #Number of features to compute
    mod = 10**(mod/10) #Convert energies in dB to original values
    n_mod_bin = n_mod_bin #Number of modulation frequency bins
    n_freq_bin = n_freq_bin #Number of conventional frequency bins
    mod = np.reshape(mod,(n_freq_bin, n_mod_bin)) #Reshape psd matrix
    ds_mod = np.empty((n_mod_bin,n_fea))*np.nan #Initialize a matrix to store descriptors in all bins
    ds_freq = np.empty((n_freq_bin,n_fea))*np.nan
    
    def get_subband_descriptors(psd, freq_range):
        #Initialize a matrix to store features
        ft=np.empty((8))*np.nan
        lo,hi = freq_range[0], freq_range[-1]#Smallest and largest value of freq_range
        
        #Centroid
        ft[0] = np.sum(psd*freq_range)/np.sum(psd)
        #Entropy
        ft[1]=-np.sum(psd*np.log(psd))/np.log(hi-lo)
        #Spread
        ft[2]=np.sqrt(np.sum(np.square(freq_range-ft[0])*psd)/np.sum(psd))
        #skewness
        ft[3]=np.sum(np.power(freq_range-ft[0],3)*psd)/(np.sum(psd)*ft[2]**3)
        #kurtosis
        ft[4]=np.sum(np.power(freq_range-ft[0],4)*psd)/(np.sum(psd)*ft[2]**4)
        #flatness
        arth_mn=np.mean(psd)/(hi-lo)
        geo_mn=np.power(np.exp(np.sum(np.log(psd))),(1/(hi-lo)))
        ft[5]=geo_mn/arth_mn
        #crest
        ft[6]=np.max(psd)/(np.sum(psd)/(hi-lo))
        #flux
        ft[7]=np.sum(np.abs(np.diff(psd)))
        
        return ft
    
    #Loop through all modulation frequency bands
    freq_bin_width = freq_lim/n_freq_bin
    mod_bin_width = mod_lim/n_mod_bin
    freq = np.arange(0,freq_lim,freq_bin_width)+freq_bin_width/2 #List of center values of frequency bins
    mod_freq = np.arange(0,mod_lim,mod_bin_width)+mod_bin_width/2 #List of center values of modulation frequency bins
    #Calculate features for each modulation frequency bin
    for mod_band in np.arange(n_mod_bin):
        ds_mod[mod_band,:] = get_subband_descriptors(mod[:,mod_band], freq)
    #Calculate features for each conventional frequency bin
    for freq_band in np.arange(n_freq_bin):
        ds_freq[freq_band,:] = get_subband_descriptors(mod[freq_band,:], mod_freq)
    
    return np.concatenate((np.reshape(ds_mod, (8*n_mod_bin)), np.reshape(ds_freq, (8*n_freq_bin))),axis=None)


def extract_msf_per_ad(ad,fs):
    
    n_fea = 8
    n_mod_bin = 20
    n_freq_bin = 20
    n_en_sample = n_mod_bin*n_freq_bin
    n_lld_sample = n_fea*n_mod_bin + n_fea*n_freq_bin
    n_fea = n_en_sample + n_lld_sample
    mod_fea = np.empty((1,n_fea))*np.nan

    mod_fea[0,:n_en_sample] = mspec_energy(ad,
                                         fs=fs,
                                         mod_lim=20,
                                         freq_lim=8000,
                                         n_mod_bin=n_mod_bin,
                                         n_f_bin=n_freq_bin)
    
    mod_fea[0,n_en_sample:] = get_mspec_descriptors(mod_fea[0,:n_en_sample],
                                                    mod_lim=20,
                                                    freq_lim=8000,
                                                    n_mod_bin=n_mod_bin,
                                                    n_freq_bin=n_freq_bin)
    
    return mod_fea



def extract_mod_fea(glued_audio_list,fs):
    
    n_samples = len(glued_audio_list)
    n_fea = 8
    n_mod_bin = 20
    n_freq_bin = 20
    n_en_sample = n_mod_bin*n_freq_bin
    n_lld_sample = n_fea*n_mod_bin + n_fea*n_freq_bin
    
    n_fea = n_en_sample + n_lld_sample
    mod_fea = np.empty((n_samples,n_fea))*np.nan
    
    for i in range(n_samples):

        mod_fea[i,:n_en_sample] = mspec_energy(glued_audio_list[i],
                                     fs=fs,
                                     mod_lim=20,
                                     freq_lim=8000,
                                     n_mod_bin=n_mod_bin,
                                     n_f_bin=n_freq_bin)
        mod_fea[i,n_en_sample:] = get_mspec_descriptors(mod_fea[i,:n_en_sample],
                                                                mod_lim=20,
                                                                freq_lim=8000,
                                                                n_mod_bin=n_mod_bin,
                                                                n_freq_bin=n_freq_bin)
    
    return mod_fea

