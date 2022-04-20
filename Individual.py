#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:40:30 2022

@author: hayun
"""
#Zed 0404
# Hi its me

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import mne
import os
import glob

bads = [['Fpz', 'AF3', 'Fc5', 'C3', 'Cp1', 'Cp5', 'T6', 'Afz', 'FT7', 'P5', 'PO3', 'C1', 'C2', 'Fz', 'FC3'],['F3', 'Fc5', 'C3', 'T3', 'Fp2', 'Cp5', 'Cp1', 'FT7', 'Afz', 'T6', 'T5', 'O1', 'F1', 'PO3', 'C1'],['F3','Fc5','C3', 'Cp5', 'Cp1', 'FT7', 'T6', 'T5', 'PO3', 'P5', 'TP8', 'C1', 'C5', 'Pz', 'P3', 'F1', 'FC3', 'F6', 'T3', 'Fc6','Fz'],['Poz', 'PO3', 'P5', 'C1', 'FT7', 'F1', 'FC3', 'Afz', 'F6', 'T5', 'T6', 'C3', 'T4', 'Cp5', 'Cp1', 'Fc5', 'F3', 'T3', 'Fc6', 'Fz', 'Pz', 'O2', 'TP8', 'Fp1', 'C5'],['F3', 'Cp5', 'Fc5', 'Cp1', 'T6', 'T5', 'FT7', 'Poz', 'C5', 'C1', 'F6', 'FC3', 'F1', 'O2', 'T3', 'T4', 'Fc6', 'Fz', 'Fp2', 'Fp1', 'C3', 'C4'],['F3', 'Fc5', 'Cp5', 'Cp1', 'Cp6', 'FT7', 'Afz', 'T6', 'T5', 'Fcz', 'C1', 'Poz', 'T3', 'Fc6', 'F1', 'FC3', 'F6', 'C5'],['Fc5', 'F3', 'Cp6', 'T4', 'Cp5', 'T6', 'Afz', 'FT7', 'Poz', 'C1', 'Fcz', 'C5', 'F1', 'FC3', 'F6', 'T3', 'Fc6', 'Fp1'],['Fc5', 'F3', 'Fz', 'T4', 'Cp6', 'FT7', 'Fcz', 'Afz', 'T6', 'C1', 'Poz', 'F1', 'C5', 'T3']]
mont = mne.channels.make_standard_montage('standard_1020')
Zed = []
for file in sorted(glob.glob('/Users/Ha-Jun/PhD/Data/Oddball/Data/Start/0404_Zed/*.EDF')):
    Zed.append(file)
low_freq = 0.1
high_freq = 30
Blocks = [[],[],[],[],[],[],[],[]]


for i in range(len(Zed)):
    Blocks[i] = mne.io.read_raw_edf(Zed[i], infer_types = True, exclude=('DIG DTRIG'), preload = True)
    Blocks[i].set_montage(mont,match_case = False, match_alias = True, on_missing = 'ignore')
    # Blocks[i].info['bads'].extend(bads[i])
    # Blocks[i].interpolate_bads(reset_bads = True, exclude = ['REF_EEG REF_EEG'])
    Blocks[i].filter(low_freq, high_freq, n_jobs = 1)
    # Blocks[i].notch_filter(freqs = np.arange(10,46,10),notch_widths = 2)
    Blocks[i].notch_filter(freqs = np.arange(50,351,50))
    # Blocks[i].set_eeg_reference(ref_channels = ['REF_EEG REF_EEG'])
    # Blocks[i].set_eeg_reference()



#%%
audio_length = 58.53061224489796
# tmin = [117.0869,132.1318,117.9798,114.0244, 112.3867,111.0732,135.8975,129.3584]
tmin = [117.0869,132.1318-0.2,117.9798,114.0244-0.3, 112.3867,111.0732+0.13,135.8975-0.2,129.3584]

tmin = np.add(tmin, 0.2)
tmax = np.add(tmin, audio_length)

#make events
duration_event = 1.3006802721088435
duration_epoch = 2
events = [0,1,2,3,4,5,0,1]
epochs = [0,1,2,3,4,5,0,1]
evoked = [0,1,2,3,4,5,0,1]

picks = ('Fcz','Fz','Fc1','Fc2')
# picks = ('P6','P5','P3','P4') 
for i in range(len(Blocks)):
    events[i] = mne.make_fixed_length_events(Blocks[i], start = tmin[i], stop = tmax[i], duration = duration_event)
    epochs[i] = mne.Epochs(Blocks[i],events[i], preload = True, tmin = -0.2 , tmax = duration_event+0.5, picks = picks)
    evoked[i] = epochs[i].average()

ML1 = [15,17,24,27,28,33,35,37,44]
MM1 = []
for i in range(45):
    if i not in ML1:
        MM1.append(i)

MR2 = [15,19,20,26,27,33,35,39,40] 
MM2 = []
for i in range(45):
    if i not in MR2:
        MM2.append(i)
        
MM3 = [16,19,23,26,27,33,36,39,43] 
ML3 = []
for i in range(45):
    if i not in MM3:
        ML3.append(i)

ML_Epoch_RC = mne.concatenate_epochs((epochs[0][ML1],epochs[2][ML3],epochs[4][MR2],epochs[6][ML1]))
MM_Epoch_RC = mne.concatenate_epochs((epochs[0][MM1],epochs[2][MM3],epochs[4][MM2],epochs[6][MM1]))


ML_Epoch_LC = mne.concatenate_epochs((epochs[1][MR2],epochs[3][ML1],epochs[5][ML3],epochs[7][MR2]))
MM_Epoch_LC = mne.concatenate_epochs((epochs[1][MM2],epochs[3][MM1],epochs[5][MM3],epochs[7][MM2]))


# #zed (1)
# RC
# ML : epochs[1][0][ML3],  epochs[1][2][ML1], epochs[1][4][MR2], epochs[1][6][ML3]
# MM : epochs[1][0][MM3],  epochs[1][2][MM1], epochs[1][4][MM2], epochs[1][6][MM3]

# LC
# ML : epochs[1][1][MR2], epochs[1][3][ML3], epochs[1][5][ML1], epochs[1][7][MR2]
# MM : epochs[1][1][MM2], epochs[1][3][MM3], epochs[1][5][MM1], epochs[1][7][MM2]


A = []
for i in range(127):
    if i%2 ==1:
        A.append(i)
        
# ML_Epoch = ML_Epoch_LC
# MM_Epoch = MM_Epoch_LC


ML_Epoch = ML_Epoch_RC
MM_Epoch = MM_Epoch_RC

ML_Epoch = mne.concatenate_epochs((ML_Epoch_LC,ML_Epoch_RC))
MM_Epoch = mne.concatenate_epochs((MM_Epoch_LC,MM_Epoch_RC))


A,B = len(ML_Epoch),len(MM_Epoch)

reject = dict(eeg = 150e-6)


ML_Epoch.drop_bad(reject = reject)
MM_Epoch.drop_bad(reject = reject)

# ML_Epoch.apply_baseline((0.4,0.6))
# MM_Epoch.apply_baseline((0.4,0.6))

ML_Epoch.apply_baseline((-0.2,0))
MM_Epoch.apply_baseline((-0.2,0))

print(A,len(ML_Epoch),B,len(MM_Epoch))


ML_Evoked = ML_Epoch.average()
MM_Evoked = MM_Epoch.average()

# picks = ('Fcz','Fz','Fc1')

mne.viz.plot_compare_evokeds(dict(MM = MM_Evoked, MLMR = ML_Evoked),colors = ['blue','red'],
                            legend='upper left', show_sensors='upper right'
                            ,combine = 'mean',title = 'MMN, RC',vlines = [0.6])

#%%

for i in range(len(Blocks)):
    events[i] = mne.make_fixed_length_events(Blocks[i], start = tmin[i], stop = tmax[i], duration = duration_event)
    epochs[i] = mne.Epochs(Blocks[i],events[i], preload = True, tmin = -0.2 , tmax = duration_event+0.5, baseline = None, picks = picks)
    evoked[i] = epochs[i].average()
    
reject = dict(eeg = 180e-6)

for i in range(8):
    epochs[i].apply_baseline((-0.2,0))
    epochs[i].drop_bad(reject = reject)

dic = dict(a = epochs[0].average(),
            b = epochs[1].average(),
            # c = epochs[2].average(),
            d = epochs[3].average(),
            e = epochs[4].average(),
            f = epochs[5].average(),
            g = epochs[6].average(),
            h = epochs[7].average()
           )

mne.viz.plot_compare_evokeds(dic,legend='upper left', show_sensors='upper right',picks=picks
                            ,combine = 'mean',title = 'RC_MM')
#%%
#Zed 0504

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import mne
import os
import glob

bads = ['F3', 'Fp2', 'F7', 'Fp1', 'Cp1', 'Cp2', 'Cp5', 'F6', 'F1', 'Fpz', 'AF3', 'T6', 'T5', 'Pz', 'O2', 'Fcz', 'CP3', 'TP8', 'P6', 'C6', 'FT8'],['F3', 'Fp2', 'Fp1', 'Fc5', 'T4', 'C4', 'Fc6', 'Cp2', 'Cp1', 'T6', 'O2', 'F1', 'F6', 'Fpz', 'AF3', 'Fcz', 'P4', 'FT8', 'CP3', 'TP8', 'PO4', 'C6', 'C5'],['F3', 'Fc2', 'Cp1', 'T6', 'P4', 'O2', 'F1', 'F6', 'Fcz', 'TP8', 'CP3', 'PO4', 'C6', 'FT8'],['F3', 'Fc2', 'Cp1', 'Fc5', 'Fc6', 'C4', 'P4', 'T6', 'O2', 'AF3', 'Fpz', 'F1', 'F6', 'Fcz', 'FT8', 'C5', 'C6', 'P6', 'PO3', 'PO4', 'TP8', 'CP3', 'P5', 'FT7'],['Fz', 'Fc2', 'Cp5', 'Cp2', 'Cp1', 'T4', 'F7', 'F8', 'F3', 'P4', 'Pz', 'T6', 'O2', 'Fpz', 'AF3', 'F1', 'F6', 'Fcz', 'T5', 'FT8', 'C5', 'P5', 'TP8', 'CP3', 'PO4', 'P6'],['F3', 'Fc2', 'Fz', 'Fc5', 'Fc6', 'T6', 'Pz',  'F1', 'Fcz', 'AF3', 'Fpz', 'F6', 'T5', 'O2', 'FT7', 'CP3', 'C6', 'TP8', 'P5', 'PO4', 'FT8', 'P6', 'Oz'],['F3', 'Fc5', 'Fc2', 'Cp1', 'T5', 'T6', 'O2', 'Fpz', 'AF3', 'F1', 'F6', 'Fcz', 'FT8', 'CP3', 'C6', 'TP8', 'P5', 'P6', 'PO4', 'FT7'],['F3', 'Fc2', 'Cp1', 'Fc1', 'C3', 'Fpz', 'AF3', 'T5', 'O2', 'T6', 'F1', 'F6', 'Fcz', 'FT8', 'CP3', 'P5', 'Oz', 'PO4', 'P6', 'TP8', 'C6']
mont = mne.channels.make_standard_montage('standard_1020')
Zed = []
for file in sorted(glob.glob('/Users/Ha-Jun/PhD/Data/Oddball/Data/Start/0504_Zed/*.EDF')):
    Zed.append(file)
low_freq = 0.1
high_freq = 30
Blocks = [[],[],[],[],[],[],[],[]]


for i in range(len(Zed)):
    Blocks[i] = mne.io.read_raw_edf(Zed[i], infer_types = True, exclude=('DIG DTRIG'), preload = True)
    Blocks[i].set_montage(mont,match_case = False, match_alias = True, on_missing = 'ignore')
    Blocks[i].info['bads'].extend(bads[i])
    Blocks[i].interpolate_bads(reset_bads = True, exclude = ['REF_EEG REF_EEG'])
    Blocks[i].filter(low_freq, high_freq, n_jobs = 1)
   # Blocks[i].notch_filter(freqs = np.arange(10,46,10),notch_widths = 2)
    Blocks[i].notch_filter(freqs = np.arange(50,351,50))
    # Blocks[i].set_eeg_reference(ref_channels = ['REF_EEG REF_EEG'])
    Blocks[i].set_eeg_reference(ref_channels = 'average')



#%%
audio_length = 58.53061224489796
tmin = [100.8984,114.7402,112.7031,100.7158,95.82617,104.1787,92.60254,129.8699]
tmin = [100.8984,114.7402,112.7031-0.15,100.7158-0.1,95.82617,104.1787,92.60254,129.8699]

tmin = np.add(tmin, 0.2)
tmax = np.add(tmin, audio_length)

#make events
duration_event = 1.3006802721088435
duration_epoch = 2
events = [0,1,2,3,4,5,0,1]
epochs = [0,1,2,3,4,5,0,1]
evoked = [0,1,2,3,4,5,0,1]

picks = ('Fcz','Fz','Fc1','Fc2')
# picks = ('P6','P5','P3','P4') 
for i in range(len(Blocks)):
    events[i] = mne.make_fixed_length_events(Blocks[i], start = tmin[i], stop = tmax[i], duration = duration_event)
    epochs[i] = mne.Epochs(Blocks[i],events[i], preload = True, tmin = -0.2 , tmax = duration_event+0.5, baseline = None, picks = picks)
    evoked[i] = epochs[i].average()

ML1 = [15,17,24,27,28,33,35,37,44]
MM1 = []
for i in range(45):
    if i not in ML1:
        MM1.append(i)

MR2 = [15,19,20,26,27,33,35,39,40] 
MM2 = []
for i in range(45):
    if i not in MR2:
        MM2.append(i)
        
MM3 = [16,19,23,26,27,33,36,39,43] 
ML3 = []
for i in range(45):
    if i not in MM3:
        ML3.append(i)

ML_Epoch_RC = mne.concatenate_epochs((epochs[0][ML1],epochs[2][ML3],epochs[4][MR2],epochs[6][ML1]))
MM_Epoch_RC = mne.concatenate_epochs((epochs[0][MM1],epochs[2][MM3],epochs[4][MM2],epochs[6][MM1]))


ML_Epoch_LC = mne.concatenate_epochs((epochs[1][MR2],epochs[3][ML1],epochs[5][ML3],epochs[7][MR2]))
MM_Epoch_LC = mne.concatenate_epochs((epochs[1][MM2],epochs[3][MM1],epochs[5][MM3],epochs[7][MM2]))


A = []
for i in range(232):
    if i%2 ==1:
        A.append(i)
        
ML_Epoch = ML_Epoch_LC
MM_Epoch = MM_Epoch_LC


# ML_Epoch = ML_Epoch_RC
# MM_Epoch = MM_Epoch_RC

# ML_Epoch = mne.concatenate_epochs((ML_Epoch_LC,ML_Epoch_RC))
# MM_Epoch = mne.concatenate_epochs((MM_Epoch_LC,MM_Epoch_RC))


A,B = len(ML_Epoch),len(MM_Epoch)

reject = dict(eeg = 150e-6)


ML_Epoch.drop_bad(reject = reject)
MM_Epoch.drop_bad(reject = reject)

# ML_Epoch.apply_baseline((0.4,0.6))
# MM_Epoch.apply_baseline((0.4,0.6))

ML_Epoch.apply_baseline((-0.2,0))
MM_Epoch.apply_baseline((-0.2,0))

print(A,len(ML_Epoch),B,len(MM_Epoch))


ML_Evoked = ML_Epoch.average()
MM_Evoked = MM_Epoch.average()

# picks = ('Fcz','Fz','Fc1')

mne.viz.plot_compare_evokeds(dict(MM = MM_Evoked, MLMR = ML_Evoked),colors = ['blue','red'],
                            legend='upper left', show_sensors='upper right'
                            ,combine = 'mean',title = 'MMN, RC',vlines = [0.6])
#%%

for i in range(len(Blocks)):
    events[i] = mne.make_fixed_length_events(Blocks[i], start = tmin[i], stop = tmax[i], duration = duration_event)
    epochs[i] = mne.Epochs(Blocks[i],events[i], preload = True, tmin = -0.2 , tmax = duration_event+0.5, baseline = None, picks = picks)
    evoked[i] = epochs[i].average()
    
reject = dict(eeg = 150e-6)

for i in range(8):
    epochs[i].apply_baseline((-0.2,0))
    epochs[i].drop_bad(reject = reject)

dic = dict(a = epochs[0].average(),
            b = epochs[1].average(),
            c = epochs[2].average(),
            d = epochs[3].average(),
            e = epochs[4].average(),
            f = epochs[5].average(),
            g = epochs[6].average(),
            h = epochs[7].average()
           )

mne.viz.plot_compare_evokeds(dic,legend='upper left', show_sensors='upper right',picks=picks
                            ,combine = 'mean',title = 'RC_MM')
#%%
#Bianca 0604

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import mne
import os
import glob

bads = [['Fc2', 'P4'],['Fc2', 'Pz', 'P4'],['Fc2', 'P4', 'C2'],['Fc2', 'F4', 'P4', 'C2', 'P2'],['Fc2', 'Afz'],['Fc5', 'Fc2', 'C2', 'Poz', 'Oz'],['Fc5', 'Fc2'],['Fc5', 'Fc2', 'F1', 'C2', 'Oz']]
mont = mne.channels.make_standard_montage('standard_1020')
Zed = []
for file in sorted(glob.glob('/Users/Ha-Jun/PhD/Data/Oddball/Data/Start/0604_Bianca/*.EDF')):
    Zed.append(file)
low_freq = 0.1
high_freq = 30
Blocks = [[],[],[],[],[],[],[],[]]


for i in range(len(Zed)):
    Blocks[i] = mne.io.read_raw_edf(Zed[i], infer_types = True, exclude=('DIG DTRIG'), preload = True)
    Blocks[i].set_montage(mont,match_case = False, match_alias = True, on_missing = 'ignore')
    # Blocks[i].info['bads'].extend(bads[i])
    # Blocks[i].interpolate_bads(reset_bads = True, exclude = ['REF_EEG REF_EEG'])
    Blocks[i].filter(low_freq, high_freq, n_jobs = 1)
   # Blocks[i].notch_filter(freqs = np.arange(10,46,10),notch_widths = 2)
    Blocks[i].notch_filter(freqs = np.arange(50,351,50))
    # Blocks[i].set_eeg_reference(ref_channels = ['REF_EEG REF_EEG'])
    # Blocks[i].set_eeg_reference(ref_channels = 'average')



#%%
audio_length = 58.53061224489796
tmin = [118.2549,138.7266,147.4556,131.6523,134.0215,165.0205,152.6768,158.624]
tmin = np.add(tmin, 0.2)
tmax = np.add(tmin, audio_length)

#make events
duration_event = 1.3006802721088435
duration_epoch = 2
events = [0,1,2,3,4,5,0,1]
epochs = [0,1,2,3,4,5,0,1]
evoked = [0,1,2,3,4,5,0,1]

picks = ('Fcz','Fz','Fc1')
# picks = ('P6','P5','P3','P4') 
for i in range(len(Blocks)):
    events[i] = mne.make_fixed_length_events(Blocks[i], start = tmin[i], stop = tmax[i], duration = duration_event)
    epochs[i] = mne.Epochs(Blocks[i],events[i], preload = True, tmin = -0.2 , tmax = duration_event+0.5, baseline = None, picks = picks)
    evoked[i] = epochs[i].average()

ML1 = [15,17,24,27,28,33,35,37,44]
MM1 = []
for i in range(45):
    if i not in ML1:
        MM1.append(i)

MR2 = [15,19,20,26,27,33,35,39,40] 
MM2 = []
for i in range(45):
    if i not in MR2:
        MM2.append(i)
        
MM3 = [16,19,23,26,27,33,36,39,43] 
ML3 = []
for i in range(45):
    if i not in MM3:
        ML3.append(i)

ML_Epoch_RC = mne.concatenate_epochs((epochs[0][ML1],epochs[2][ML3],epochs[4][MR2],epochs[6][ML1]))
MM_Epoch_RC = mne.concatenate_epochs((epochs[0][MM1],epochs[2][MM3],epochs[4][MM2],epochs[6][MM1]))


ML_Epoch_LC = mne.concatenate_epochs((epochs[1][MR2],epochs[3][ML1],epochs[5][ML3],epochs[7][MR2]))
MM_Epoch_LC = mne.concatenate_epochs((epochs[1][MM2],epochs[3][MM1],epochs[5][MM3],epochs[7][MM2]))


A = []
for i in range(232):
    if i%2 ==1:
        A.append(i)
        
ML_Epoch = ML_Epoch_LC
MM_Epoch = MM_Epoch_LC


# ML_Epoch = ML_Epoch_RC
# MM_Epoch = MM_Epoch_RC

# ML_Epoch = mne.concatenate_epochs((ML_Epoch_LC,ML_Epoch_RC))
# MM_Epoch = mne.concatenate_epochs((MM_Epoch_LC,MM_Epoch_RC))[A]


A,B = len(ML_Epoch),len(MM_Epoch)

reject = dict(eeg = 150e-6)


ML_Epoch.drop_bad(reject = reject)
MM_Epoch.drop_bad(reject = reject)

# ML_Epoch.apply_baseline((0.4,0.6))
# MM_Epoch.apply_baseline((0.4,0.6))

# ML_Epoch.apply_baseline((-0.2,0))
# MM_Epoch.apply_baseline((-0.2,0))

print(A,len(ML_Epoch),B,len(MM_Epoch))


ML_Evoked = ML_Epoch.average()
MM_Evoked = MM_Epoch.average()

# picks = ('Fcz','Fz','Fc1')

mne.viz.plot_compare_evokeds(dict(MM = MM_Evoked, MLMR = ML_Evoked),colors = ['blue','red'],
                            legend='upper left', show_sensors='upper right'
                            ,combine = 'mean',title = 'MMN, RC',vlines = [0.6])

#%%
#Sam 0604

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import mne
import os
import glob

bads = [['Fc6', 'T3', 'Cp5', 'T6', 'T5', 'Fcz', 'C5'],['Fc6', 'T3', 'Cp5', 'T6', 'T5', 'Fcz', 'C5', 'Fp1'],['T3', 'Cp5', 'Fc6', 'T5', 'T6', 'C5', 'Fcz', 'P2', 'O1', 'Fp1'],['T3', 'Cp5', 'Fc6', 'T6', 'T5', 'Fcz', 'FC3', 'P6', 'C5'],['Fc6', 'Cp5', 'T3', 'T6', 'T5', 'FC3', 'P6', 'C5', 'Fcz'],['Fc6', 'T3', 'Cp5', 'T5', 'T6', 'FC3', 'Fcz', 'P6', 'C5'],['Fc6', 'T3', 'Cp5', 'T6', 'T5', 'FC3', 'Fcz', 'P6', 'C5'],['Fc6', 'T3', 'Cp5', 'FC3', 'T6', 'T5', 'C5', 'Fcz', 'C2', 'Cpz', 'P6', 'P2']]
mont = mne.channels.make_standard_montage('standard_1020')
Zed = []
for file in sorted(glob.glob('/Users/Ha-Jun/PhD/Data/Oddball/Data/Start/0604_Sam/*.EDF')):
    Zed.append(file)
low_freq = 0.1
high_freq = 30
Blocks = [[],[],[],[],[],[],[],[]]


for i in range(len(Zed)):
    Blocks[i] = mne.io.read_raw_edf(Zed[i], infer_types = True, exclude=('DIG DTRIG'), preload = True)
    Blocks[i].set_montage(mont,match_case = False, match_alias = True, on_missing = 'ignore')
    Blocks[i].info['bads'].extend(bads[i])
    Blocks[i].interpolate_bads(reset_bads = True, exclude = ['REF_EEG REF_EEG'])
    Blocks[i].filter(low_freq, high_freq, n_jobs = 1)
   # Blocks[i].notch_filter(freqs = np.arange(10,46,10),notch_widths = 2)
    Blocks[i].notch_filter(freqs = np.arange(50,351,50))
    Blocks[i].set_eeg_reference(ref_channels = ['REF_EEG REF_EEG'])
    Blocks[i].set_eeg_reference(ref_channels = 'average')



#%%
audio_length = 58.53061224489796
tmin = [127.1689,122.5322,124.9473,133.8721,140.6592,101.54,143.585,213.3066]
#tmin = np.add(tmin, -0.8)
tmax = np.add(tmin, audio_length)

#make events
duration_event = 1.3006802721088435
duration_epoch = 2
events = [0,1,2,3,4,5,0,1]
epochs = [0,1,2,3,4,5,0,1]
evoked = [0,1,2,3,4,5,0,1]

picks = ('Fcz','Fz','Fc1','Fc2')
# picks = ('P6','P5','P3','P4') 
for i in range(len(Blocks)):
    events[i] = mne.make_fixed_length_events(Blocks[i], start = tmin[i], stop = tmax[i], duration = duration_event)
    epochs[i] = mne.Epochs(Blocks[i],events[i], preload = True, tmin = -0.2 , tmax = duration_event+0.5, baseline = None, picks = picks)
    evoked[i] = epochs[i].average()

ML1 = [15,17,24,27,28,33,35,37,44]
MM1 = []
for i in range(45):
    if i not in ML1:
        MM1.append(i)

MR2 = [15,19,20,26,27,33,35,39,40] 
MM2 = []
for i in range(45):
    if i not in MR2:
        MM2.append(i)
        
MM3 = [16,19,23,26,27,33,36,39,43] 
ML3 = []
for i in range(45):
    if i not in MM3:
        ML3.append(i)

MR4 = [15,20,22,26,28,33,35,40,42]
ML4 = []
for i in range(45):
    if i not in MR4:
        ML4.append(i)

MM5 = [16,18,22,27,30,33,36,38,42]
MR5 = []
for i in range(45):
    if i not in MM5:
        MR5.append(i)

ML_Epoch_RC = mne.concatenate_epochs((epochs[0][ML1],epochs[2][ML3],epochs[4][MR2],epochs[6][ML1]))
MM_Epoch_RC = mne.concatenate_epochs((epochs[0][MM1],epochs[2][MM3],epochs[4][MM2],epochs[6][MM1]))


ML_Epoch_LC = mne.concatenate_epochs((epochs[1][MR2],epochs[3][ML1],epochs[5][ML3],epochs[7][MR2]))
MM_Epoch_LC = mne.concatenate_epochs((epochs[1][MM2],epochs[3][MM1],epochs[5][MM3],epochs[7][MM2]))


A = []
for i in range(232):
    if i%2 ==1:
        A.append(i)
        
# ML_Epoch = ML_Epoch_LC
# MM_Epoch = MM_Epoch_LC


ML_Epoch = ML_Epoch_RC
MM_Epoch = MM_Epoch_RC

ML_Epoch = mne.concatenate_epochs((ML_Epoch_LC,ML_Epoch_RC))
MM_Epoch = mne.concatenate_epochs((MM_Epoch_LC,MM_Epoch_RC))[A]


A,B = len(ML_Epoch),len(MM_Epoch)

reject = dict(eeg = 150e-6)


ML_Epoch.drop_bad(reject = reject)
MM_Epoch.drop_bad(reject = reject)

# ML_Epoch.apply_baseline((0.4,0.6))
# MM_Epoch.apply_baseline((0.4,0.6))

# ML_Epoch.apply_baseline((-0.2,0))
# MM_Epoch.apply_baseline((-0.2,0))

print(A,len(ML_Epoch),B,len(MM_Epoch))


ML_Evoked = ML_Epoch.average()
MM_Evoked = MM_Epoch.average()

# picks = ('Fcz','Fz','Fc1')

mne.viz.plot_compare_evokeds(dict(MM = MM_Evoked, MLMR = ML_Evoked),colors = ['blue','red'],
                            legend='upper left', show_sensors='upper right'
                            ,combine = 'mean',title = 'MMN, RC',vlines = [0.6])

#%%
#Joseph 0804

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import mne
import os
import glob

bads = [['P3', 'P1', 'P2', 'P4'],['P4', 'P3', 'Pz', 'P1', 'P2'],['P3', 'P4', 'P1', 'P2'],['P4', 'P3', 'T6', 'P1', 'P2'],['P4', 'P3', 'P1', 'P2'],['P3', 'P4', 'P1', 'P2', 'P6'],['P3', 'P4', 'P1', 'P2'],['Fc1', 'Cp1', 'P4', 'P3', 'F2', 'P2', 'P1', 'P6', 'C2']]
mont = mne.channels.make_standard_montage('standard_1020')
Zed = []
for file in sorted(glob.glob('/Users/Ha-Jun/PhD/Data/Oddball/Data/Start/0804_Joseph/*.EDF')):
    Zed.append(file)
low_freq = 0.1
high_freq = 30
Blocks = [[],[],[],[],[],[],[],[]]


for i in range(len(Zed)):
    Blocks[i] = mne.io.read_raw_edf(Zed[i], infer_types = True, exclude=('DIG DTRIG','Fpz','AF3'), preload = True)
    Blocks[i].set_montage(mont,match_case = False, match_alias = True, on_missing = 'ignore')
    Blocks[i].info['bads'].extend(bads[i])
    Blocks[i].interpolate_bads(reset_bads = True, exclude = ['REF_EEG REF_EEG'])
    Blocks[i].filter(low_freq, high_freq, n_jobs = 2)
    # Blocks[i].notch_filter(freqs = np.arange(10,46,10),notch_widths = 2)
    Blocks[i].notch_filter(freqs = np.arange(50,351,50))
    # Blocks[i].set_eeg_reference(ref_channels = ['REF_EEG REF_EEG'])
    # Blocks[i].set_eeg_reference(ref_channels = 'average')



#%%
audio_length = 58.53061224489796
tmin = [126.209,138.3096,120.2705,107.252,117.6836,142.5889,140.2363,122.9834]
tmin = np.add(tmin, 0.5)
tmax = np.add(tmin, audio_length)

#make events
duration_event = 1.3006802721088435
duration_epoch = 2
events = [0,1,2,3,4,5,0,1]
epochs = [0,1,2,3,4,5,0,1]
evoked = [0,1,2,3,4,5,0,1]

picks = ('Fcz','Fz','Fc1')
# picks = ('P6','P5','P3','P4') 
for i in range(len(Blocks)):
    events[i] = mne.make_fixed_length_events(Blocks[i], start = tmin[i], stop = tmax[i], duration = duration_event)
    epochs[i] = mne.Epochs(Blocks[i],events[i], preload = True, tmin = -0.2 , tmax = duration_event, baseline = None, picks = picks)
    evoked[i] = epochs[i].average()

ML1 = [15,17,24,27,28,33,35,37,44]
MM1 = []
for i in range(45):
    if i not in ML1:
        MM1.append(i)

MR2 = [15,19,20,26,27,33,35,39,40] 
MM2 = []
for i in range(45):
    if i not in MR2:
        MM2.append(i)
        
MM3 = [16,19,23,26,27,33,36,39,43] 
ML3 = []
for i in range(45):
    if i not in MM3:
        ML3.append(i)

MR4 = [15,20,22,26,28,33,35,40,42]
ML4 = []
for i in range(45):
    if i not in MR4:
        ML4.append(i)

MM5 = [16,18,22,27,30,33,36,38,42]
MR5 = []
for i in range(45):
    if i not in MM5:
        MR5.append(i)

ML_Epoch_RC = mne.concatenate_epochs((epochs[0][ML1],  epochs[1][MR5], epochs[4][ML4],epochs[4][MR4], epochs[5][ML1]))
MM_Epoch_RC = mne.concatenate_epochs((epochs[0][MM1],  epochs[1][MM5], epochs[5][MM1]))


ML_Epoch_LC = mne.concatenate_epochs((epochs[2][MR4], epochs[2][ML4], epochs[3][MR5], epochs[6][ML1], epochs[7][ML4], epochs[7][MR4]))
MM_Epoch_LC = mne.concatenate_epochs((epochs[3][MM5], epochs[6][MM1]))




A = []
for i in range(120):
    if i%3 ==1:
        A.append(i)
        
ML_Epoch = ML_Epoch_LC
MM_Epoch = MM_Epoch_LC


# ML_Epoch = ML_Epoch_RC
# MM_Epoch = MM_Epoch_RC

# ML_Epoch = mne.concatenate_epochs((ML_Epoch_LC,ML_Epoch_RC))
# MM_Epoch = mne.concatenate_epochs((MM_Epoch_LC,MM_Epoch_RC))


A,B = len(ML_Epoch),len(MM_Epoch)

reject = dict(eeg = 150e-6)


ML_Epoch.drop_bad(reject = reject)
MM_Epoch.drop_bad(reject = reject)

# ML_Epoch.apply_baseline((0.4,0.6))
# MM_Epoch.apply_baseline((0.4,0.6))

# ML_Epoch.apply_baseline((-0.2,0))
# MM_Epoch.apply_baseline((-0.2,0))

print(A,len(ML_Epoch),B,len(MM_Epoch))


ML_Evoked = ML_Epoch.average()
MM_Evoked = MM_Epoch.average()

# picks = ('Fcz','Fz','Fc1')

mne.viz.plot_compare_evokeds(dict(MM = MM_Evoked, MLMR = ML_Evoked),colors = ['blue','red'],
                            legend='upper left', show_sensors='upper right'
                            ,combine = 'mean',title = 'MMN, RC',vlines = [0.6])

#%%

for i in range(len(Blocks)):
    events[i] = mne.make_fixed_length_events(Blocks[i], start = tmin[i], stop = tmax[i], duration = duration_event)
    epochs[i] = mne.Epochs(Blocks[i],events[i], preload = True, tmin = 0 , tmax = duration_event, baseline = None, picks = picks)
    evoked[i] = epochs[i].average()
    
reject = dict(eeg = 180e-6)

for i in range(8):
    # epochs[i].apply_baseline((-0.2,0))
    epochs[i].drop_bad(reject = reject)

dic = dict(a = epochs[0].average(),
            b = epochs[1].average(),
            c = epochs[2].average(),
            d = epochs[3].average(),
            # e = epochs[4].average(),
            # f = epochs[5].average(),
            # g = epochs[6].average(),
            # h = epochs[7].average()
           )

mne.viz.plot_compare_evokeds(dic,legend='upper left', show_sensors='upper right',picks=picks
                            ,combine = 'mean',title = 'RC_MM')

#%%
#Kiaran 1104

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import mne
import os
import glob

bads = [['Cp6', 'Fcz', 'Oz', 'PO3', 'FT8', 'T6', 'Fz'],['Fz', 'F4', 'Cp6', 'Fcz', 'FT8', 'PO3', 'Oz'],['Fz', 'F4', 'Cp2', 'Cp6', 'Fcz', 'Oz', 'FT8', 'TP8', 'PO3'],['F4', 'Cp2', 'Cp6', 'Fcz', 'TP8', 'FT8', 'Oz', 'Fz'],['F4', 'Fz', 'Cp2', 'Cp6', 'P4', 'Fcz', 'TP8', 'FT8', 'Oz', 'C4'],['Fz', 'F4', 'Cp6', 'C4', 'T6', 'O1', 'P4', 'Pz', 'FT8', 'Fcz', 'C6', 'TP8', 'Oz'],['Fz', 'F4', 'Fc6', 'C3', 'C4', 'Cp6', 'O1', 'P4', 'Pz', 'F2', 'Fcz', 'FT8', 'TP8', 'PO4', 'Oz', 'O2'],['Fp1', 'Fz', 'Fc6', 'C3', 'C4', 'Cp6', 'P4', 'Fcz', 'O1', 'O2', 'FT8', 'TP8', 'PO4', 'Oz', 'F4', 'Fc5']]
mont = mne.channels.make_standard_montage('standard_1020')
Zed = []
for file in sorted(glob.glob('/Users/Ha-Jun/PhD/Data/Oddball/Data/Start/1104_Kiaran/*.EDF')):
    Zed.append(file)
low_freq = 0.1
high_freq = 30
Blocks = [[],[],[],[],[],[],[],[]]


for i in range(len(Zed)):
    Blocks[i] = mne.io.read_raw_edf(Zed[i], infer_types = True, exclude=('DIG DTRIG'), preload = True)
    Blocks[i].set_montage(mont,match_case = False, match_alias = True, on_missing = 'ignore')
    Blocks[i].info['bads'].extend(bads[i])
    Blocks[i].interpolate_bads(reset_bads = True, exclude = ['REF_EEG REF_EEG'])
    Blocks[i].filter(low_freq, high_freq, n_jobs = 1)
   # Blocks[i].notch_filter(freqs = np.arange(10,46,10),notch_widths = 2)
    Blocks[i].notch_filter(freqs = np.arange(50,351,50))
    Blocks[i].set_eeg_reference(ref_channels = ['REF_EEG REF_EEG'])
    Blocks[i].set_eeg_reference(ref_channels = 'average')



#%%
audio_length = 58.53061224489796
tmin = [179.4463,153.2422,154.9121,107.6494,125.0986,182.4375,119.3086,126.6797]
#tmin = np.add(tmin, -0.8)
tmax = np.add(tmin, audio_length)

#make events
duration_event = 1.3006802721088435
duration_epoch = 2
events = [0,1,2,3,4,5,0,1]
epochs = [0,1,2,3,4,5,0,1]
evoked = [0,1,2,3,4,5,0,1]

picks = ('Fcz','Fz','Fc1','Fc2')
# picks = ('P6','P5','P3','P4') 
for i in range(len(Blocks)):
    events[i] = mne.make_fixed_length_events(Blocks[i], start = tmin[i], stop = tmax[i], duration = duration_event)
    epochs[i] = mne.Epochs(Blocks[i],events[i], preload = True, tmin = -0.2 , tmax = duration_event+0.5, baseline = None, picks = picks)
    evoked[i] = epochs[i].average()

ML1 = [15,17,24,27,28,33,35,37,44]
MM1 = []
for i in range(45):
    if i not in ML1:
        MM1.append(i)

MR2 = [15,19,20,26,27,33,35,39,40] 
MM2 = []
for i in range(45):
    if i not in MR2:
        MM2.append(i)
        
MM3 = [16,19,23,26,27,33,36,39,43] 
ML3 = []
for i in range(45):
    if i not in MM3:
        ML3.append(i)

ML_Epoch_RC = mne.concatenate_epochs((epochs[0][ML1],epochs[2][ML3],epochs[4][MR2],epochs[6][ML1]))
MM_Epoch_RC = mne.concatenate_epochs((epochs[0][MM1],epochs[2][MM3],epochs[4][MM2],epochs[6][MM1]))


ML_Epoch_LC = mne.concatenate_epochs((epochs[1][MR2],epochs[3][ML1],epochs[5][ML3],epochs[7][MR2]))
MM_Epoch_LC = mne.concatenate_epochs((epochs[1][MM2],epochs[3][MM1],epochs[5][MM3],epochs[7][MM2]))


A = []
for i in range(232):
    if i%2 ==1:
        A.append(i)
        
# ML_Epoch = ML_Epoch_LC
# MM_Epoch = MM_Epoch_LC


ML_Epoch = ML_Epoch_RC
MM_Epoch = MM_Epoch_RC

ML_Epoch = mne.concatenate_epochs((ML_Epoch_LC,ML_Epoch_RC))
MM_Epoch = mne.concatenate_epochs((MM_Epoch_LC,MM_Epoch_RC))[A]


A,B = len(ML_Epoch),len(MM_Epoch)

reject = dict(eeg = 150e-6)


ML_Epoch.drop_bad(reject = reject)
MM_Epoch.drop_bad(reject = reject)

# ML_Epoch.apply_baseline((0.4,0.6))
# MM_Epoch.apply_baseline((0.4,0.6))

# ML_Epoch.apply_baseline((-0.2,0))
# MM_Epoch.apply_baseline((-0.2,0))

print(A,len(ML_Epoch),B,len(MM_Epoch))


ML_Evoked = ML_Epoch.average()
MM_Evoked = MM_Epoch.average()

# picks = ('Fcz','Fz','Fc1')

mne.viz.plot_compare_evokeds(dict(MM = MM_Evoked, MLMR = ML_Evoked),colors = ['blue','red'],
                            legend='upper left', show_sensors='upper right'
                            ,combine = 'mean',title = 'MMN, RC',vlines = [0.6])

#%%
#Daniel 1204

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import mne
import os
import glob

bads = [['F4'],['F4', 'Pz', 'FT8', 'C2'],['F4', 'T4', 'Cp6', 'FT8', 'C2'],['F4', 'FT8', 'C2'],['F4', 'Pz', 'FT8', 'F6'],['F4', 'Pz', 'O2', 'F6', 'AF3', 'Fpz', 'O1', 'FT7', 'FC4', 'FT8', 'C2'],['F4', 'Pz', 'T5', 'F6', 'C6'],['F4', 'Pz', 'F6', 'C6', 'FT8']]
mont = mne.channels.make_standard_montage('standard_1020')
Zed = []
for file in sorted(glob.glob('/Users/Ha-Jun/PhD/Data/Oddball/Data/Start/1204_Daniel/*.EDF')):
    Zed.append(file)
low_freq = 0.1
high_freq = 30
Blocks = [[],[],[],[],[],[],[],[]]


for i in range(len(Zed)):
    Blocks[i] = mne.io.read_raw_edf(Zed[i], infer_types = True, exclude=('DIG DTRIG'), preload = True)
    Blocks[i].set_montage(mont,match_case = False, match_alias = True, on_missing = 'ignore')
    Blocks[i].info['bads'].extend(bads[i])
    Blocks[i].interpolate_bads(reset_bads = True, exclude = ['REF_EEG REF_EEG'])
    Blocks[i].filter(low_freq, high_freq, n_jobs = 1)
   # Blocks[i].notch_filter(freqs = np.arange(10,46,10),notch_widths = 2)
    Blocks[i].notch_filter(freqs = np.arange(50,351,50))
    Blocks[i].set_eeg_reference(ref_channels = ['REF_EEG REF_EEG'])
    Blocks[i].set_eeg_reference(ref_channels = 'average')



#%%
audio_length = 58.53061224489796
tmin = [124.6582,148.2725,201.3896,128.0654,186.3418,151.9609,144.6358,147.7803]
#tmin = np.add(tmin, -0.8)
tmax = np.add(tmin, audio_length)

#make events
duration_event = 1.3006802721088435
duration_epoch = 2
events = [0,1,2,3,4,5,0,1]
epochs = [0,1,2,3,4,5,0,1]
evoked = [0,1,2,3,4,5,0,1]

picks = ('Fcz','Fz','Fc1','Fc2')
# picks = ('P6','P5','P3','P4') 
for i in range(len(Blocks)):
    events[i] = mne.make_fixed_length_events(Blocks[i], start = tmin[i], stop = tmax[i], duration = duration_event)
    epochs[i] = mne.Epochs(Blocks[i],events[i], preload = True, tmin = -0.2 , tmax = duration_event+0.5, baseline = None, picks = picks)
    evoked[i] = epochs[i].average()

ML1 = [15,17,24,27,28,33,35,37,44]
MM1 = []
for i in range(45):
    if i not in ML1:
        MM1.append(i)

MR2 = [15,19,20,26,27,33,35,39,40] 
MM2 = []
for i in range(45):
    if i not in MR2:
        MM2.append(i)
        
MM3 = [16,19,23,26,27,33,36,39,43] 
ML3 = []
for i in range(45):
    if i not in MM3:
        ML3.append(i)

ML_Epoch_RC = mne.concatenate_epochs((epochs[0][ML1],epochs[2][ML3],epochs[4][MR2],epochs[6][ML1]))
MM_Epoch_RC = mne.concatenate_epochs((epochs[0][MM1],epochs[2][MM3],epochs[4][MM2],epochs[6][MM1]))


ML_Epoch_LC = mne.concatenate_epochs((epochs[1][MR2],epochs[3][ML1],epochs[5][ML3],epochs[7][MR2]))
MM_Epoch_LC = mne.concatenate_epochs((epochs[1][MM2],epochs[3][MM1],epochs[5][MM3],epochs[7][MM2]))


A = []
for i in range(232):
    if i%2 ==1:
        A.append(i)
        
# ML_Epoch = ML_Epoch_LC
# MM_Epoch = MM_Epoch_LC


ML_Epoch = ML_Epoch_RC
MM_Epoch = MM_Epoch_RC

ML_Epoch = mne.concatenate_epochs((ML_Epoch_LC,ML_Epoch_RC))
MM_Epoch = mne.concatenate_epochs((MM_Epoch_LC,MM_Epoch_RC))[A]


A,B = len(ML_Epoch),len(MM_Epoch)

reject = dict(eeg = 150e-6)


ML_Epoch.drop_bad(reject = reject)
MM_Epoch.drop_bad(reject = reject)

# ML_Epoch.apply_baseline((0.4,0.6))
# MM_Epoch.apply_baseline((0.4,0.6))

# ML_Epoch.apply_baseline((-0.2,0))
# MM_Epoch.apply_baseline((-0.2,0))

print(A,len(ML_Epoch),B,len(MM_Epoch))


ML_Evoked = ML_Epoch.average()
MM_Evoked = MM_Epoch.average()

# picks = ('Fcz','Fz','Fc1')

mne.viz.plot_compare_evokeds(dict(MM = MM_Evoked, MLMR = ML_Evoked),colors = ['blue','red'],
                            legend='upper left', show_sensors='upper right'
                            ,combine = 'mean',title = 'MMN, RC',vlines = [0.6]),

#%%
#Zoe 1204

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import mne
import os
import glob

bads = [['C4', 'T4', 'C3', 'T3', 'Cp6', 'Cp2', 'O1', 'T6', 'P4', 'P3', 'T5', 'C5', 'C6', 'Cpz', 'TP8', 'P5', 'P1', 'P2', 'PO3', 'Poz', 'PO4'],['T3', 'C3', 'C4', 'T4', 'Cp2', 'Cp6', 'O1', 'T5', 'P3', 'P4', 'T6', 'C5', 'C6', 'Cpz', 'TP8', 'P5', 'P2', 'P1', 'PO3', 'Poz', 'PO4'],['F4', 'T3', 'C3', 'C4', 'T4', 'Cp2', 'Cp6', 'T6', 'P4', 'T5', 'C5', 'C6', 'Cpz', 'TP8', 'P1', 'P5', 'PO3', 'P6', 'P2'],['F4', 'T3', 'C3', 'C4', 'T4', 'Cp2', 'Cp6', 'P4', 'T6', 'T5', 'C5', 'C6', 'Cpz', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO3'],['F4', 'T3', 'C3', 'C4', 'T4', 'Cp2', 'Cp6', 'T6', 'P4', 'T5', 'C6', 'C5', 'Cpz', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO3'],['F4', 'T3', 'C3', 'C4', 'T4', 'Cp2', 'Cp6', 'P4', 'T5', 'T6', 'C5', 'C6', 'Cpz', 'TP8', 'P1', 'P2', 'P5', 'P6', 'PO3', 'Oz'],['F4', 'T3', 'C3', 'C4', 'T4', 'Cp2', 'Cp6', 'Cp5', 'T6', 'P4', 'T5', 'C5', 'C6', 'TP8', 'P1', 'P5', 'P2', 'P6', 'PO3', 'Cpz'],['F4', 'T3', 'Fc5', 'T4', 'C3', 'C4', 'Cp2', 'Cp6', 'P4', 'T5', 'T6', 'C5', 'C6', 'CP3', 'Cpz', 'P5', 'TP8', 'P1', 'P2', 'P6', 'PO3', 'Oz']]
mont = mne.channels.make_standard_montage('standard_1020')
Zed = []
for file in sorted(glob.glob('/Users/Ha-Jun/PhD/Data/Oddball/Data/Start/1204_Zoe/*.EDF')):
    Zed.append(file)
low_freq = 0.1
high_freq = 30
Blocks = [[],[],[],[],[],[],[],[]]


for i in range(len(Zed)):
    Blocks[i] = mne.io.read_raw_edf(Zed[i], infer_types = True, exclude=('DIG DTRIG'), preload = True)
    Blocks[i].set_montage(mont,match_case = False, match_alias = True, on_missing = 'ignore')
    # Blocks[i].info['bads'].extend(bads[i])
    # Blocks[i].interpolate_bads(reset_bads = True, exclude = ['REF_EEG REF_EEG'])
    Blocks[i].filter(low_freq, high_freq, n_jobs = 1)
   # Blocks[i].notch_filter(freqs = np.arange(10,46,10),notch_widths = 2)
    Blocks[i].notch_filter(freqs = np.arange(50,351,50))
    Blocks[i].set_eeg_reference(ref_channels = ['REF_EEG REF_EEG'])
    # Blocks[i].set_eeg_reference(ref_channels = 'average')



#%%
audio_length = 58.53061224489796
tmin = [149.4209,148.8867,151.4492,184.126,150.2764,150.5879,146.7744,151.7402]
#tmin = np.add(tmin, -0.8)
tmax = np.add(tmin, audio_length)

#make events
duration_event = 1.3006802721088435
duration_epoch = 2
events = [0,1,2,3,4,5,0,1]
epochs = [0,1,2,3,4,5,0,1]
evoked = [0,1,2,3,4,5,0,1]

picks = ('Fcz','Fz','Fc1','Fc2')
# picks = ('P6','P5','P3','P4') 
for i in range(len(Blocks)):
    events[i] = mne.make_fixed_length_events(Blocks[i], start = tmin[i], stop = tmax[i], duration = duration_event)
    epochs[i] = mne.Epochs(Blocks[i],events[i], preload = True, tmin = -0.2 , tmax = duration_event+0.5, baseline = None, picks = picks)
    evoked[i] = epochs[i].average()

ML1 = [15,17,24,27,28,33,35,37,44]
MM1 = []
for i in range(45):
    if i not in ML1:
        MM1.append(i)

MR2 = [15,19,20,26,27,33,35,39,40] 
MM2 = []
for i in range(45):
    if i not in MR2:
        MM2.append(i)
        
MM3 = [16,19,23,26,27,33,36,39,43] 
ML3 = []
for i in range(45):
    if i not in MM3:
        ML3.append(i)

ML_Epoch_RC = mne.concatenate_epochs((epochs[0][ML1],epochs[2][ML3],epochs[4][MR2],epochs[6][ML1]))
MM_Epoch_RC = mne.concatenate_epochs((epochs[0][MM1],epochs[2][MM3],epochs[4][MM2],epochs[6][MM1]))


ML_Epoch_LC = mne.concatenate_epochs((epochs[1][MR2],epochs[3][ML1],epochs[5][ML3],epochs[7][MR2]))
MM_Epoch_LC = mne.concatenate_epochs((epochs[1][MM2],epochs[3][MM1],epochs[5][MM3],epochs[7][MM2]))


A = []
for i in range(232):
    if i%2 ==1:
        A.append(i)
        
# ML_Epoch = ML_Epoch_LC
# MM_Epoch = MM_Epoch_LC


ML_Epoch = ML_Epoch_RC
MM_Epoch = MM_Epoch_RC

ML_Epoch = mne.concatenate_epochs((ML_Epoch_LC,ML_Epoch_RC))
MM_Epoch = mne.concatenate_epochs((MM_Epoch_LC,MM_Epoch_RC))[A]


A,B = len(ML_Epoch),len(MM_Epoch)

reject = dict(eeg = 150e-6)


ML_Epoch.drop_bad(reject = reject)
MM_Epoch.drop_bad(reject = reject)

# ML_Epoch.apply_baseline((0.4,0.6))
# MM_Epoch.apply_baseline((0.4,0.6))

# ML_Epoch.apply_baseline((-0.2,0))
# MM_Epoch.apply_baseline((-0.2,0))

print(A,len(ML_Epoch),B,len(MM_Epoch))


ML_Evoked = ML_Epoch.average()
MM_Evoked = MM_Epoch.average()

# picks = ('Fcz','Fz','Fc1')

mne.viz.plot_compare_evokeds(dict(MM = MM_Evoked, MLMR = ML_Evoked),colors = ['blue','red'],
                            legend='upper left', show_sensors='upper right'
                            ,combine = 'mean',title = 'MMN, RC',vlines = [0.6])

#%%
#Daniel 1304

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import mne
import os
import glob

bads = [['C2'],['F3', 'Pz', 'C6'],['F3', 'C6', 'FC4'],['F3', 'Fc6', 'C3', 'Afz', 'AF4', 'C6'],['F3', 'T5', 'Afz', 'AF4'],['F3', 'Cp2', 'Afz', 'AF4', 'T5'],['C3', 'F3', 'Cp2', 'Afz', 'T5', 'AF4', 'C6', 'CP4'],['F3', 'C3', 'Cp2', 'Afz', 'AF4', 'T5', 'C6', 'CP4']]
mont = mne.channels.make_standard_montage('standard_1020')
Zed = []
for file in sorted(glob.glob('/Users/Ha-Jun/PhD/Data/Oddball/Data/Start/1304_Daniel/*.EDF')):
    Zed.append(file)
low_freq = 0.1
high_freq = 30
Blocks = [[],[],[],[],[],[],[],[]]


for i in range(len(Zed)):
    Blocks[i] = mne.io.read_raw_edf(Zed[i], infer_types = True, exclude=('DIG DTRIG'), preload = True)
    Blocks[i].set_montage(mont,match_case = False, match_alias = True, on_missing = 'ignore')
    # Blocks[i].info['bads'].extend(bads[i])
    # Blocks[i].interpolate_bads(reset_bads = True, exclude = ['REF_EEG REF_EEG'])
    Blocks[i].filter(low_freq, high_freq, n_jobs = 1)
   # Blocks[i].notch_filter(freqs = np.arange(10,46,10),notch_widths = 2)
    Blocks[i].notch_filter(freqs = np.arange(50,351,50))
    Blocks[i].set_eeg_reference(ref_channels = ['REF_EEG REF_EEG'])
    # Blocks[i].set_eeg_reference(ref_channels = 'average')



#%%
audio_length = 58.53061224489796
tmin = [122.6797,185.4026,146.2402,128.4736,176.2256,242.7168,182.7246,198.8857]
#tmin = np.add(tmin, -0.8)
tmax = np.add(tmin, audio_length)

#make events
duration_event = 1.3006802721088435
duration_epoch = 2
events = [0,1,2,3,4,5,0,1]
epochs = [0,1,2,3,4,5,0,1]
evoked = [0,1,2,3,4,5,0,1]

picks = ('Fcz','Fz','Fc1','Fc2')
# picks = ('P6','P5','P3','P4') 
for i in range(len(Blocks)):
    events[i] = mne.make_fixed_length_events(Blocks[i], start = tmin[i], stop = tmax[i], duration = duration_event)
    epochs[i] = mne.Epochs(Blocks[i],events[i], preload = True, tmin = -0.2 , tmax = duration_event+0.5, baseline = None, picks = picks)
    evoked[i] = epochs[i].average()

ML1 = [15,17,24,27,28,33,35,37,44]
MM1 = []
for i in range(45):
    if i not in ML1:
        MM1.append(i)

MR2 = [15,19,20,26,27,33,35,39,40] 
MM2 = []
for i in range(45):
    if i not in MR2:
        MM2.append(i)
        
MM3 = [16,19,23,26,27,33,36,39,43] 
ML3 = []
for i in range(45):
    if i not in MM3:
        ML3.append(i)

ML_Epoch_RC = mne.concatenate_epochs((epochs[0][ML1],epochs[2][ML3],epochs[4][MR2],epochs[6][ML1]))
MM_Epoch_RC = mne.concatenate_epochs((epochs[0][MM1],epochs[2][MM3],epochs[4][MM2],epochs[6][MM1]))


ML_Epoch_LC = mne.concatenate_epochs((epochs[1][MR2],epochs[3][ML1],epochs[5][ML3],epochs[7][MR2]))
MM_Epoch_LC = mne.concatenate_epochs((epochs[1][MM2],epochs[3][MM1],epochs[5][MM3],epochs[7][MM2]))


A = []
for i in range(232):
    if i%2 ==1:
        A.append(i)
        
# ML_Epoch = ML_Epoch_LC
# MM_Epoch = MM_Epoch_LC


ML_Epoch = ML_Epoch_RC
MM_Epoch = MM_Epoch_RC

ML_Epoch = mne.concatenate_epochs((ML_Epoch_LC,ML_Epoch_RC))
MM_Epoch = mne.concatenate_epochs((MM_Epoch_LC,MM_Epoch_RC))[A]


A,B = len(ML_Epoch),len(MM_Epoch)

reject = dict(eeg = 150e-6)


ML_Epoch.drop_bad(reject = reject)
MM_Epoch.drop_bad(reject = reject)

# ML_Epoch.apply_baseline((0.4,0.6))
# MM_Epoch.apply_baseline((0.4,0.6))

# ML_Epoch.apply_baseline((-0.2,0))
# MM_Epoch.apply_baseline((-0.2,0))

print(A,len(ML_Epoch),B,len(MM_Epoch))


ML_Evoked = ML_Epoch.average()
MM_Evoked = MM_Epoch.average()

# picks = ('Fcz','Fz','Fc1')

mne.viz.plot_compare_evokeds(dict(MM = MM_Evoked, MLMR = ML_Evoked),colors = ['blue','red'],
                            legend='upper left', show_sensors='upper right'
                            ,combine = 'mean',title = 'MMN, RC',vlines = [0.6])