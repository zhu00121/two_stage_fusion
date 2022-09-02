# TASLP2022-Fusion-of-MSF-and-LP-features-for-COVID19-detection
Scripts for replicating results shown in our TASLP 2022 paper 'Fusion of modulation spectrogram and linear prediction features for COVID-19 detection'

## Repository structure
-- Repo
  -- feature: stores extracted modulation features and LP features
  -- script
    -- ```LPmain.py```: Extract LP features from DICOVA2 dataset
    -- ```Cambridge_LPmain.py```: Extract LP features from Cambridge dataset (track2)
    -- ```LPfunc.py```: Functions for LP analysis and feature extraction
    -- ```two_stage.py```: Functions to build a two-stage classification system
    -- ```feature_eva.py```: Main code to evaluate feature performance
