#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 18:26:31 2024

@author: Team D
"""

import os
import numpy as np
import mne
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from mne.decoding import cross_val_multiscore


sample_data_folder = mne.datasets.sample.data_path()  # Setting the path to MNE sample data
sample_data_evk_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis-ave.fif')  # Constructing the path to the evoked file

# Reading evoked data from a file, applying baseline correction, and storing in a list
evokeds_list = mne.read_evokeds(sample_data_evk_file, baseline=(None, 0), proj=True, verbose=False)

# Show the condition names and baseline correction status for each evoked dataset
for e in evokeds_list:
    print(f'Condition: {e.comment}, baseline: {e.baseline}')
    
conds = ('aud/left', 'aud/right', 'vis/left', 'vis/right')
evks = dict(zip(conds, evokeds_list))  # Creating a dictionary of evoked datasets with condition names as keys

# Plotting evoked data for 'aud/left' condition
evks['aud/left'].plot(exclude=[])

# Plotting magnetometer data for 'aud/left' condition with spatial colors and global field power
evks['aud/left'].plot(picks='mag', spatial_colors=True, gfp=True)

times = np.linspace(0.05, 0.13, 5)
# Plotting topographic maps for magnetometer data of 'aud/left' condition at specified times
evks['aud/left'].plot_topomap(ch_type='mag', times=times, colorbar=True)

# Plotting arrow maps for magnetometer data of 'aud/left' condition
mags = evks['aud/left'].copy().pick_types(meg='mag')
mne.viz.plot_arrowmap(mags.data[:, 175], mags.info, extrapolate='local')

# Plotting joint plot for evoked data of 'vis/right' condition
evks['vis/right'].plot_joint()

# Plotting comparison of evoked datasets with different combination methods
def custom_func(x):
    return x.max(axis=1)

for combine in ('mean', 'median', 'gfp', custom_func):
    mne.viz.plot_compare_evokeds(evks, picks='eeg', combine=combine)
    
# Plotting comparison of evoked datasets for a specific MEG channel
mne.viz.plot_compare_evokeds(evks, picks='MEG 1811', colors=dict(aud=0, vis=1), linestyles=dict(left='solid', right='dashed'))

temp_list = list()
# Creating a temporary list of evoked datasets with modified comments
for idx, _comment in enumerate(('foo', 'foo', '', None, 'bar'), start=1):
    _evk = evokeds_list[0].copy()
    _evk.comment = _comment
    _evk.data *= idx  # Multiplying data to differentiate traces
    temp_list.append(_evk)
    
# Plotting comparison of temporary evoked datasets
mne.viz.plot_compare_evokeds(temp_list, picks='mag')

# Plotting image of evoked data for 'vis/right' condition
evks['vis/right'].plot_image(picks='meg')

# Plotting comparison of evoked datasets with customization options for EEG channels
mne.viz.plot_compare_evokeds(evks, picks='eeg', colors=dict(aud=0, vis=1), linestyles=dict(left='solid', right='dashed'), axes='topo', styles=dict(aud=dict(linewidth=1), vis=dict(linewidth=1)))

# Plotting topographic maps of evoked data for all conditions
mne.viz.plot_evoked_topo(evokeds_list)

subjects_dir = os.path.join(sample_data_folder, 'subjects')
sample_data_trans_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis_raw-trans.fif')

# Making and plotting field maps for 'aud/left' condition
maps = mne.make_field_map(evks['aud/left'], trans=sample_data_trans_file, subject='sample', subjects_dir=subjects_dir)
evks['aud/left'].plot_field(maps, time=0.1)

# Making and plotting field maps for each channel type
for ch_type in ('mag', 'grad', 'eeg'):
    evk = evks['aud/right'].copy().pick(ch_type)
    _map = mne.make_field_map(evk, trans=sample_data_trans_file, subject='sample', subjects_dir=subjects_dir, meg_surf='head')
    fig = evk.plot_field(_map, time=0.1)
    mne.viz.set_3d_title(fig, ch_type, size=20)
    
# Evaluation

# Setting the path to MNE sample data
sample_data_folder = mne.datasets.sample.data_path()

# Constructing the path to the evoked file
sample_data_evk_file = os.path.join(sample_data_folder, 'MEG', 'sample', 'sample_audvis-ave.fif')

# Reading evoked data from a file, applying baseline correction, and storing in a list
evokeds_list = mne.read_evokeds(sample_data_evk_file, baseline=(None, 0), proj=True, verbose=False)

# Creating a dictionary of evoked datasets with condition names as keys
conds = ('aud/left', 'aud/right', 'vis/left', 'vis/right')
evks = dict(zip(conds, evokeds_list))

# Define features (X) and labels (y) for classification
X = []
y = []
for cond_name, evk in evks.items():
    X.append(evk.data)  # Using raw evoked data as features
    if 'aud' in cond_name:
        y.append(0)  # Assigning label 0 for auditory conditions
    else:
        y.append(1)  # Assigning label 1 for visual conditions
X = np.array(X)
y = np.array(y)

# Flatten the data for RandomForestClassifier
X_flattened = np.concatenate([x.reshape(1, -1) for x in X])

# Define the machine learning model (Random Forest Classifier)
estimator = RandomForestClassifier()

# Perform leave-one-out cross-validation (LOOCV) and evaluate the model's performance
loo = LeaveOneOut()
scores = cross_val_multiscore(estimator, X_flattened, y, cv=loo)

# Print the cross-validation scores and mean accuracy
print("Cross-validation scores:", scores)
print("Mean accuracy:", np.mean(scores))