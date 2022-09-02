# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 17:29:29 2022

@author: richa
"""

import os
import LPfunc
import argparse, configparser


def lp_main(file_info, **kwargs):
    
    print('---Start Processing---')
    print(file_info.case)
    
    if (file_info.case == 0) or (file_info.case == 2):
        
        if not os.path.exists(file_info.ad_original_path):
            raise Exception('folder path (original) does not exist!')
        
        print('Removing silence frames...')
        """ remove silence from original audio files """
        LPfunc.process_ad_folder(file_info.ad_original_path,
                                 file_info.ad_clean_path,
                                 file_info.audio_type_og,
                                 file_info.sr)
        
        print('Silence removed.')
    
    if (file_info.case == 1) or (file_info.case == 2):
        
        if not os.path.exists(file_info.ad_clean_path):
            raise Exception('folder path (clean) does not exist!')
        
        print('Extracting features...')
        """ extract LP features """
        lp_group = LPfunc.extract_lp_in_group(file_info.ad_clean_path,
                                              file_info.audio_type_clean,
                                              file_info.sr, 
                                              **kwargs['DEFAULT'])
        print('LP feature extracted.')
        
        """ save LP features """
        LPfunc.save_data(file_info.outdir,
                         file_info.outname,
                         lp_group)
        print('Feature saved.')
        
    return 0


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, const='./LP_param.ini', nargs='?', help='path to lp configuration file')
    parser.add_argument('--case', type=int, const=0, nargs='?', help='--0:remove silence; --1:feature extraction; --2:both')
    parser.add_argument('--audio_type_og', type=str, const='flac', nargs='?', help='Audio type: .wav/.flac/etc.')
    parser.add_argument('--audio_type_clean', type=str, const='flac', nargs='?', help='Audio type: .wav/.flac/etc.')
    parser.add_argument('--sr', type=int, const=16000, nargs='?', help='sampling rate')
    parser.add_argument('--ad_original_path', type=str, required=False, help='original path of audio files')
    parser.add_argument('--ad_clean_path',type=str, required=False, help='path of audio files without silence periods')
    parser.add_argument('--outdir',type=str, required=False)
    parser.add_argument('--outname', type=str, required=False)
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    # config['DEFAULT'] = {'normalize': 'no',
    #                      'lp_order': 18}
    # with open('./LP_param.ini', 'w') as configfile:
    #     config.write(configfile)
    config.read(args.config)
    
    """ check errors in input """
    if ((args.case == 0 or args.case == 2) and \
        'ad_original_path' not in vars(args)):
        
        parser.error('--case:0/1 requires the --ad_original_path')

    if ((args.case == 1 or args.case == 2) and \
        'ad_clean_path' not in vars(args)):
        
        parser.error('--case:1/2 requires the --ad_clean_path')

    """ execute """
    lp_main(args,**config)