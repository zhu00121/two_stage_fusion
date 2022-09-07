# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:11:11 2022

@author: richa
"""

import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, PredefinedSplit
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

""" Some functions used for model training, evaluation, etc."""
def data_preprocess(x_train, x_valid, x_test):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_valid = scaler.fit_transform(x_valid)
    X_test = scaler.transform(x_test)
    
    return X_train, X_valid, X_test


def train_model(X_train,y_train,
                model:str,
                clf_kwargs):
    
    """ choose model """
    model_choice = ['svm','rf','dt']
    assert model in model_choice, " available models: svm/rf/dt "
    
    if model == 'svm':
        m = SVC(kernel = 'linear',probability=True)
        params = {'C':[0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.007,0.01,0.05,0.1,0.5,1]}
    
    elif model == 'rf':
        m = RandomForestClassifier()
        params = {'n_estimators': [5,10,30,50,100,200],
                 'max_depth': [2, 5, 10, 20]}
    
    if clf_kwargs['pds'] == 'no':
        """ N-fold stratefied cross-validation """
        cv = RepeatedStratifiedKFold(n_splits=clf_kwargs["n_splits"], n_repeats=clf_kwargs["n_repeats"], random_state=26)
    
    elif clf_kwargs['pds'] == 'yes':
        cv = PredefinedSplit(clf_kwargs['split_index'])
        
    clf = GridSearchCV(m, param_grid=params,cv=cv,scoring='roc_auc')
    clf.fit(X_train,y_train)
    print (clf.best_params_)

    return clf


def get_confmat(label,prediction):
    cf_matrix = confusion_matrix(label,prediction)

    group_names = ['True neg','False pos','False neg','True pos']
    group_percentages1 = ["{0:.2%}".format(value) for value in
                          cf_matrix[0]/np.sum(cf_matrix[0])]
    group_percentages2 = ["{0:.2%}".format(value) for value in
                          cf_matrix[1]/np.sum(cf_matrix[1])]

    group_percentages = group_percentages1+group_percentages2
    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2,v3 in
              zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    yticklabels = ['negative','positive']
    xticklabels = ['negative','positive']
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues',yticklabels=yticklabels,
                xticklabels=xticklabels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    
    return cf_matrix


def per_stage(x_train,y_train,
              x_valid,y_valid,
              x_test,y_test,
              model:str,
              clf_kwargs):
    
    """ preprocess """
    X_train, X_valid, X_test = data_preprocess(x_train,x_valid,x_test)
    
    split_index = [-1]*len(X_train) + [0]*len(X_valid)
    X = np.concatenate((X_train, X_valid), axis=0)
    y = np.concatenate((y_train, y_valid), axis=0)
    
    """ train model """
    if clf_kwargs['pds'] == 'yes':
        clf_kwargs['split_index'] = split_index
        
    clf = train_model(X,y,model,clf_kwargs)
    
    """ get COVID and non-COVID predictions """
    preds = clf.predict(X_test)
    pos_idx = np.where(preds==1)[0] 
    neg_idx = np.where(preds==1)[0] # will be sent to stage2
    
    return clf,pos_idx,neg_idx


def two_stage_main(data:dict,clf_kwargs):
    
    """
    Stage1: Find COVID samples with abnormalities in artiulators using 
    modulation spectrogram features.
    """
    clf_1,pos_idx_1,neg_idx_1 = per_stage(data['msf_train'],
                                          data['label_train'],
                                          data['msf_valid'],
                                          data['label_valid'],
                                          data['msf_test'],
                                          data['label_test'],
                                          'svm',
                                          clf_kwargs)
    
    """ 
    Stage2: Within negative predictions from stage1, select those with 
    abnormalities in phonation system using LP features.
    """
    clf_2,pos_idx_2,neg_idx_2 = per_stage(data['lp_train'],
                                          data['label_train'],
                                          data['lp_valid'],
                                          data['label_valid'],
                                          data['lp_test'][neg_idx_1,:],
                                          data['label_test'][neg_idx_1,:],
                                          'rf',
                                          clf_kwargs)
    
    """ Aggregate predictions together. Need to be careful with indexing. """
    preds_final = np.full_like(data['label_test'], 0)
    preds_final[pos_idx_1] = 1
    preds_final[neg_idx_1[pos_idx_2]] = 1
    
    return preds_final


def sys_evaluate(preds,probs,label):
    
    UAR = sklearn.metrics.recall_score(label, preds, average='macro')
    ROC = sklearn.metrics.roc_auc_score(label, probs)
    
    # plt.figure(dpi=400)
    # cf = get_confmat(label,preds)
    
    report = {'UAR':UAR, 'ROC':ROC}
    
    return report
    

