# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:12:22 2022

@author: richa
"""


import os
import joblib
import librosa
import numpy as np
import pandas as pd
import LPfunc
import Modulation_spectrogram_feature as msf
import argparse, configparser
from tqdm import tqdm


def prep_mod(user_path:str,
             meta_path:str,
             track:str,
             fold:str,
             fs:int,
             feature_path:str):
    
    """ input metadata (with label) """
    df = pd.read_csv(os.path.join(user_path,meta_path))
    ids = list(df.loc[df['fold']==fold].index)
    num_files = len(ids)
    ot = np.empty((num_files,721))
    
    """ Modulation feature processing starts """
    print('Prepare %s %s data'%(track,fold))
    for k,i in tqdm(enumerate(ids)):
        
        print('--------')
        file = df.iloc[i]
        ad_path = os.path.join(user_path,'0426_EN_used_task2',file['%s_path'%(track)])
        lab = file['label']
        print('processing %s th file; user_id: %s'%(k, file['uid']))
        
        """ load data """
        print('load data...')
        ad_data = LPfunc.load_one_ad(ad_name=ad_path,fs=fs)
        
        """ extract MSF """
        print('extracting MSF...')
        mod_fea = msf.extract_msf_per_ad(ad_data,fs) 
        
        lab = lab.reshape((1,1))
        mod_final = np.concatenate((mod_fea,lab),axis=1)
        ot[k,:] = mod_final
    
    """ save MSF """
    feature_name = 'Cam_msf_%s'%(fold)
    LPfunc.save_data(feature_path,feature_name,ot)
    
    return 0


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--fs', type=int, const=16000, nargs='?', help='sampling rate')
    parser.add_argument('--user_path', type=str, required=True, help='user parent path of the Cambridge dataset folder')
    parser.add_argument('--meta_path',type=str, required=True, help='path of Cambridge dataset metadata')
    parser.add_argument('--outdir',type=str, required=True, help='output features saving path')
    parser.add_argument('--track', type=str, required=True, help='track:breathing/cough/voice')
    parser.add_argument('--fold', type=str, required=True, help='fold:train/validation/test')
    args = parser.parse_args()
    
    
    prep_mod(args.user_path,
             args.meta_path,
             args.track,
             args.fold,
             args.fs,
             args.outdir)

        