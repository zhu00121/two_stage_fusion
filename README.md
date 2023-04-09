## Physiology-inspired two-stage COVID-19 detection system
This repository contains scripts used to produce reults shown in our IEEE-TASLP paper 'COVID-19 Detection via Fusion of Modulation Spectrum and Linear Prediction Speech Features'.

Link to the paper: https://ieeexplore.ieee.org/document/10097559 (early-access version).

### Hand-crafted features
We provided two types of feature sets to capture different abnormalities of articulation system: <br />
1. **Modulation spectrum features** <br />
We also have another repository which includes a toolbox that automatically extracts different versions of modulation spectrum, it has been used previously for characterizing unnatural speech, evaluating reverberation level, and many other applications in biomedical signal analysis. You can find the toolbox here: <https://github.com/MuSAELab/modulation_filterbanks>

2. **Linear Prediction (LP) features** <br />
LP analysis has been used for separating excitation source and vocal tract, we further extracted low-level descriptors from LP residuals to characterize abnoarmal phonation pattern.

### System overview
<img src="https://user-images.githubusercontent.com/48067384/230748724-df1abe7f-e93f-4291-aff1-ed16d7e2175b.jpg" width="600" height="200">

### Repository structure
- feature: stores extracted modulation features and LP features <br />
- script <br />
   - ```LPfunc.py```: Functions for LP analysis and feature extraction <br />
   - ```Dico_LPmain.py```: Extract LP features from DICOVA2 dataset <br />
   - ```Cambridge_LPmain.py```: Extract LP features from Cambridge dataset (track2) <br />
   - ```Dico_MODmain.py```: Extract modulation spectrogram features from DICOVA2 dataset <br />
   - ```Cambridge_MODmain.py```: Extract modulation spectrogram features from Cambridge dataset (track2) <br />
   - ```two_stage.py```: Functions to build a two-stage classification system <br />
   - ```feature_eva.py```: Main code to evaluate feature performance <br />

### Data availability
COVID-19 datasets experimented in our study can be obtained upon requests from the data holders. Please reach out to them to get access these datasets.

### Citation
If you find our feature sets or the system useful, you may use the following format to cite our paper:
```
@ARTICLE{10097559,
  author={Zhu, Yi and Tiwari, Abhishek and Monteiro, João and Kshirsagar, Shruti and Falk, Tiago H.},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
  title={COVID-19 Detection via Fusion of Modulation Spectrum and Linear Prediction Speech Features}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TASLP.2023.3265603}}
  ```
