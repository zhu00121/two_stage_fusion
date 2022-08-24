# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 13:11:11 2022

@author: richa
"""

import numpy as np
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.svm import SVC,LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

""" Some functions used for model training, evaluation, etc."""
def data_preprocess(x_train,y_train,
                    x_test,y_test):
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)
    
    return X_train, y_train, X_test, y_test


def train_model(X_train,y_train,
                model:str,
                n_splits=3,n_repeats=3):
    
    """ choose model """
    model_choice = ['svm','rf','dt']
    assert model in model_choice, " available models: svm/rf/dt "
    
    if model == 'svm':
        model = SVC(kernel = 'linear',probability=True)
    
    """ N-fold stratefied cross-validation """
    params = {'C':[0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.007,0.01,0.05,0.1,0.5,1]}
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=26)
    clf = GridSearchCV(model, param_grid=params,cv=cv,scoring='roc_auc')
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
              x_test,y_test,
              model:str,
              n_splits=3,n_repeats=3):
    
    """ preprocess """
    X_train, y_train, X_test, y_test = data_preprocess(x_train,y_train,
                                                       x_test,y_test)
    
    """ train model """
    clf = train_model(X_train,y_train,
                      model=model,
                      n_splits=n_splits,
                      n_repeats=n_repeats)
    
    """ get COVID and non-COVID predictions """
    preds = clf.predict(X_test)
    pos_idx = np.where(preds==1)[0] 
    neg_idx = np.where(preds==1)[0] # will be sent to stage2
    
    return clf,pos_idx,neg_idx


def two_stage_main(msf_path,
                   lp_path,
                   dataset:str):
    
    """ load extracted features and labels """
    data = {}
    data['msf_train'] = 0
    data['msf_test'] = 0
    data['lp_train'] = 0
    data['lp_test'] = 0
    data['label_train'] = 0
    data['label_test'] = 0
    
    """
    Stage1: Find COVID samples with abnormalities in artiulators using 
    modulation spectrogram features.
    """
    clf_1,pos_idx_1,neg_idx_1 = per_stage(data['msf_train'],
                                          data['label_train'],
                                          data['msf_test'],
                                          data['label_test'],
                                          model = 'svm')
    
    """ 
    Stage2: Within negative predictions from stage1, select those with 
    abnormalities in phonation system using LP features.
    """
    clf_2,pos_idx_2,neg_idx_2 = per_stage(data['lp_train'],
                                          data['label_train'],
                                          data['lp_test'][neg_idx_1,:],
                                          data['label_test'][neg_idx_1,:],
                                          model = 'rf')
    
    """ Aggregate predictions together. Need to be careful with indexing. """
    preds_final = np.full_like(data['label_test'], 0)
    preds_final[pos_idx_1] = 1
    preds_final[neg_idx_1[pos_idx_2]] = 1
    
    return preds_final


def sys_evaluate(preds,label):
    
    UAR = sklearn.metrics.recall_score(label, preds, average='macro')
    
    plt.figure(dpi=400)
    cf = get_confmat(label,preds)
    
    report = {'UAR':UAR,'cf':cf}
    
    return report
    

