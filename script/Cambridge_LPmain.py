# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 00:14:39 2022

@author: richa
"""


import os
import joblib
import librosa
import numpy as np
import pandas as pd
import LPfunc
import argparse, configparser
from tqdm import tqdm


def prep_fea(user_path:str,
             meta_path:str,
             track:str,
             fold:str,
             fs:int,
             feature_path:str,
             lp_param:dict):
    
    """ input metadata (with label) """
    df = pd.read_csv(os.path.join(user_path,meta_path))
    ids = list(df.loc[df['fold']==fold].index)
    num_files = len(ids)
    ot = np.empty((num_files,37))
    
    """ LP feature processing starts """
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
        
        """ remove silence frames """
        print('removing silence...')
        new_data = LPfunc.remove_silence(signal=ad_data,fs=fs)
        
        """ save edited audio to the same path as original audio """
        uppath = lambda _path, n: os.sep.join(_path.split(os.sep)[:-n])
        save_folder_path = uppath(ad_path,1) # get folder path
        LPfunc.save_one_ad(folder_path=save_folder_path,
                           ad_name='audio_file_read_clean.wav',
                           ad=new_data,
                           fs=fs)
        
        """ extract LP features """
        print('extracting LP features...')
        lp_fea = LPfunc.extract_lp(os.path.join(save_folder_path,'audio_file_read_clean.wav'),
                                   **lp_param['DEFAULT']) 
        
        lab = lab.reshape((1,1))
        lp_final = np.concatenate((lp_fea,lab),axis=1)
        ot[k,:] = lp_final
    
    """ save LP features """
    feature_name = 'Cam_lp_%s'%(fold)
    LPfunc.save_data(feature_path,feature_name,ot)
    
    return 0


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, const='./LP_param.ini', nargs='?', help='path to lp configuration file')
    parser.add_argument('--fs', type=int, const=16000, nargs='?', help='sampling rate')
    parser.add_argument('--user_path', type=str, required=True, help='user parent path of the Cambridge dataset folder')
    parser.add_argument('--meta_path',type=str, required=True, help='path of Cambridge dataset metadata')
    parser.add_argument('--outdir',type=str, required=True, help='output features saving path')
    parser.add_argument('--track', type=str, required=True, help='track:breathing/cough/voice')
    parser.add_argument('--fold', type=str, required=True, help='fold:train/validation/test')
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)
    
    prep_fea(args.user_path,
             args.meta_path,
             args.track,
             args.fold,
             args.fs,
             args.outdir,
             config)

        