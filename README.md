# TASLP2022-Fusion-of-MSF-and-LP-features-for-COVID19-detection
Scripts for replicating results shown in our TASLP 2022 paper 'Fusion of modulation spectrogram and linear prediction features for COVID-19 detection'

## Repository structure
- feature: stores extracted modulation features and LP features <br />
 - script <br />
   - ```LPfunc.py```: Functions for LP analysis and feature extraction <br />
   - ```Dico_LPmain.py```: Extract LP features from DICOVA2 dataset <br />
   - ```Cambridge_LPmain.py```: Extract LP features from Cambridge dataset (track2) <br />
   - ```Dico_MODmain.py```: Extract modulation spectrogram features from DICOVA2 dataset <br />
   - ```Cambridge_MODmain.py```: Extract modulation spectrogram features from Cambridge dataset (track2) <br />
   - ```two_stage.py```: Functions to build a two-stage classification system <br />
   - ```feature_eva.py```: Main code to evaluate feature performance <br />
