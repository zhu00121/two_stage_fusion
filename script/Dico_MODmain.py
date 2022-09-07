# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 22:43:57 2022

@author: richa
"""

import os
import glob
import numpy as np
import LPfunc
import Modulation_spectrogram_feature as msf
import argparse, configparser
import fnmatch
from tqdm import tqdm


def mod_main(file_info):
    
    print('---Start Processing---')
    
    if not os.path.exists(file_info.ad_path):
        raise Exception('folder path (original) does not exist!')
    
    print('Extracting features...')
    
    num_ad = len(fnmatch.filter(os.listdir(file_info.ad_path), '*.%s'%(file_info.ad_type)))
    mod_group = np.zeros((num_ad,720))
    
    for i, file in tqdm(enumerate(sorted(glob.glob(os.path.join(file_info.ad_path, '*.%s'%(file_info.ad_type)))))):
    
        ad_data = LPfunc.load_one_ad(ad_name=file,fs=file_info.sr)
        
        # just in case some audios could be very short
        if len(ad_data)>file_info.sr:
            mod_fea = msf.extract_msf_per_ad(ad_data,file_info.sr)
            mod_group[i,:] = mod_fea
    
    print('MSF extracted.')
        
    """ save LP features """
    LPfunc.save_data(file_info.outdir,
                     file_info.outname,
                     mod_group)
    print('Feature saved.')
        
    return 0


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ad_type', type=str, const='flac', nargs='?', help='Audio type: .wav/.flac/etc.')
    parser.add_argument('--sr', type=int, const=16000, nargs='?', help='sampling rate')
    parser.add_argument('--ad_path', type=str, required=True, help='original path of audio files')
    parser.add_argument('--outdir',type=str, required=True)
    parser.add_argument('--outname', type=str, required=True)
    args = parser.parse_args()

    """ execute """
    mod_main(args)