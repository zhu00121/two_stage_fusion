# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:01:07 2022

@author: richa
"""

import os
import pickle
import two_stage
import argparse, configparser


def load_data(feature_path:str, label=False):
    
    with open(feature_path, 'rb') as f:
        data = pickle.load(f)
    
    if label:
        data = data[:,-1]
    if not label:
        data = data[:,:-1]
        
    return data

def load_sets(system_info,
              fea:str):
    
    set_tr = load_data(os.path.join(system_info.feature_path,'%s_%s_train.pkl'%(system_info.dataset,fea)))
    set_vl = load_data(os.path.join(system_info.feature_path,'%s_%s_validation.pkl'%(system_info.dataset,fea)))
    set_ts = load_data(os.path.join(system_info.feature_path,'%s_%s_test.pkl'%(system_info.dataset,fea)))
    
    return set_tr,set_vl,set_ts

    
def load_label(system_info,
               fea:str):
    
    set_tr = load_data(os.path.join(system_info.feature_path,'%s_%s_train.pkl'%(system_info.dataset,fea)),label=True)
    set_vl = load_data(os.path.join(system_info.feature_path,'%s_%s_validation.pkl'%(system_info.dataset,fea)),label=True)
    set_ts = load_data(os.path.join(system_info.feature_path,'%s_%s_test.pkl'%(system_info.dataset,fea)),label=True)
    
    return set_tr,set_vl,set_ts
    
    
def feature_evaluate(system_info,clf_info):
    
    data = {}
    
    if system_info.case == 2:
        
        """ get data """
        data['msf_train'],data['msf_valid'],data['msf_test'] = load_sets(system_info,
                                                                         fea='msf')
        
        data['lp_train'],data['lp_valid'],data['lp_test'] = load_sets(system_info,
                                                                      fea='lp')

        data['label_train'],data['label_valid'],data['label_test'] = load_label(system_info,
                                                                                fea='lp')
        
        """ train a two-stage classifier """
        print('training classifier...')
        preds = two_stage.two_stage_main(data,clf_info)
        report = two_stage.sys_evaluate(preds, data['label_test'])
        print('Test UAR score is '+str(report['UAR']))
        
    elif system_info.case == 1:
        
        data['lp_train'],data['lp_valid'],data['lp_test'] = load_sets(system_info,
                                                                      fea='lp')

        data['label_train'],data['label_valid'],data['label_test'] = load_label(system_info,
                                                                                fea='lp')
        
        """ train classifier """
        print('training classifier...')
        clf,_,_ = two_stage.per_stage(data['lp_train'], data['label_train'], 
                                      data['lp_valid'], data['label_valid'],
                                      data['lp_test'], data['label_test'], 
                                      'rf',
                                      clf_info)
    
        preds_train = clf.predict(data['lp_train'])
        preds_test = clf.predict(data['lp_test'])
        report_train = two_stage.sys_evaluate(preds_train, data['label_train'])
        report_test = two_stage.sys_evaluate(preds_test, data['label_test'])
        
        print('Train UAR score is '+str(report_train['UAR']))
        print('Test UAR score is '+str(report_test['UAR']))
        
    elif system_info.case == 0:
        
        data['msf_train'],data['msf_valid'],data['msf_test'] = load_sets(system_info,
                                                                         fea='msf')
        
        data['label_train'],data['label_valid'],data['label_test'] = load_label(system_info,
                                                                                fea='lp')

        """ train classifier """
        print('training classifier...')
        clf,_,_ = two_stage.per_stage(data['msf_train'], data['label_train'], 
                                      data['msf_valid'], data['label_valid'],
                                      data['msf_test'], data['label_test'], 
                                      'svm',
                                      clf_info)
        
        preds_train = clf.predict(data['msf_train'])
        preds_test = clf.predict(data['msf_test'])
        report_train = two_stage.sys_evaluate(preds_train, data['label_train'])
        report_test = two_stage.sys_evaluate(preds_test, data['label_test'])
    
        print('Train UAR score is '+str(report_train['UAR']))
        print('Test UAR score is '+str(report_test['UAR']))
    
    return 0




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, required=True, help='choose which feature to use')
    parser.add_argument('--feature_path', type=str, required=True, help='path to the folder which stores all feature .pkl files')
    parser.add_argument('--dataset', type=str, required=True, help='name of dataset: Dico/Cam')
    # parser.add_argument('--clf_config', type=str, const='./clf_config.ini', nargs='?', help='path to classification configuration file')
    args = parser.parse_args()
    
    clf_info = {'pds':'no',
              'split_index':[],
              'n_splits':3,
              'n_repeats':3}
    
    feature_evaluate(args,clf_info)