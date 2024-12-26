import mne 

import numpy as np 
import pandas as pd 
import os
from tqdm import tqdm
import glob
from matplotlib import pyplot as plt


#to get the dataframe records from the dataset
def get_dataframe_records(df, name, trial_number, matching_condition, channel_list):
    df_record = df[df['name'].eq(name) & df['trial number'].eq(trial_number) & df['matching condition'].eq(matching_condition)].set_index(['sensor position']).loc[channel_list]
    return df_record

#the function to get the signal array for visualization
def get_signal_array(df, name, trial_number, matching_condition, channel_list):
    df_record = df[df['name'].eq(name) & df['trial number'].eq(trial_number) & df['matching condition'].eq(matching_condition)].set_index(['sensor position']).loc[channel_list]
    return df_record.to_numpy()[:, 4:]


#The function to plot the topomap for the eeg data
def plot_topomap(signal_array, save_path_animation=None, show_names=False, start_time=0.05, end_time=1, step_size=0.1):

    montage = mne.channels.make_standard_montage('standard_1020')
    
    ch_to_remove = []
    for ch in channel_list_fixed:
        if ch not in list(set(montage.ch_names).intersection(channel_list_fixed)):
            ch_to_remove.append(channel_list_fixed.index(ch))
    arr = np.delete(signal_array.copy(), ch_to_remove, axis=0)
    
    info = mne.create_info(ch_names=list(set(montage.ch_names).intersection(channel_list_fixed)), sfreq=256, ch_types='eeg')
    
    evkd = mne.EvokedArray(arr, info)
    
    evkd.set_montage(montage)

    evkd.plot_topomap(np.arange(start_time, end_time, step_size), ch_type='eeg', time_unit='s', ncols=5, nrows=2, show_names=show_names)
    

#The function to plot the jointed topomap for the eeg data    
def plot_joint_topomap(signal_array, save_path_animation=None, show_names=False, start_time=0.05, end_time=1, step_size=0.1):

    montage = mne.channels.make_standard_montage('standard_1020')
    
    ch_to_remove = []
    for ch in channel_list_fixed:
        if ch not in list(set(montage.ch_names).intersection(channel_list_fixed)):
            ch_to_remove.append(channel_list_fixed.index(ch))
    arr = np.delete(signal_array.copy(), ch_to_remove, axis=0)
    
    info = mne.create_info(ch_names=list(set(montage.ch_names).intersection(channel_list_fixed)), sfreq=256, ch_types='eeg')
    evkd = mne.EvokedArray(arr, info)
    
    evkd.set_montage(montage)

    evkd.plot_joint()
        
sample_data_folder = mne.datasets.sample.data_path() 
sample_data_evk_file = os.path.join(sample_data_folder, 'MEG', 'sample',
                                    'sample_audvis-ave.fif')  


#Load the csv file into dataframe
_dfs_list = []
p = glob.glob('SMNI_CMI_TRAIN/*.csv')
for files in tqdm(glob.glob('SMNI_CMI_TRAIN/*.csv')):
    _dfs_list.append(pd.read_csv(files))
print(_dfs_list)
df = pd.concat(_dfs_list)
del(_dfs_list)
df = df.drop(['Unnamed: 0'], axis=1)
df.head(3)

channel_list = list(set(df['sensor position']))
channel_list.sort()


#The dictionary to correct the channel name
channel_mapping_dict = {
    'AFZ':'AFz', 
    'CPZ':'CPz', 
    'CZ':'Cz', 
    'FCZ':'FCz', 
    'FP1':'Fp1',
    'FP2':'Fp2', 
    'FPZ':'Fpz', 
    'FZ':'Fz', 
    'OZ':'Oz', 
    'POZ':'POz', 
    'PZ':'Pz',
}

channel_mapping_full = dict()


#map the channel names
for ch in channel_list:
    if ch in channel_mapping_dict:
        channel_mapping_full[ch] = channel_mapping_dict[ch]
    else:
        channel_mapping_full[ch] = ch

channel_list_fixed = [channel_mapping_full[ch] for ch in channel_list]
        
df['sensor position'] = df['sensor position'].map(channel_mapping_full)
df.head(3)

transposed_dataframe_list = []

#organize and reconstruct the dataframe containing EEG data
for group_dataframe in tqdm(df.groupby(['name', 'trial number', 'matching condition', 'sensor position', 'subject identifier'])):
    tmp = pd.DataFrame(group_dataframe[1]['sensor value']).T
    tmp.columns = [f'sample_{idx}' for idx in range(256)]
    tmp['name'] = group_dataframe[0][0]
    tmp['trial number'] = group_dataframe[0][1]
    tmp['matching condition'] = group_dataframe[0][2]
    tmp['sensor position'] = group_dataframe[0][3]
    tmp['subject identifier'] = group_dataframe[0][4]
    
    transposed_dataframe_list.append(tmp)
    
df = pd.concat(transposed_dataframe_list)
df = df[[*df.columns[-5:],*df.columns[0:-5]]]
df = df.reset_index(drop=True)
df.head(3)


#visualize the dataset
df_record = get_dataframe_records(df, 'co2a0000364', 0, 'S1 obj', channel_list_fixed)


signal_array = get_signal_array(df, 'co2a0000364', 10, 'S1 obj', channel_list_fixed)

plt.title('Signal Array as an image (64 x 256)')
plt.ylabel('Sensor Position)')
plt.xlabel('Sample Numbers')
plt.imshow(signal_array.astype(int))
plt.show()


#generate plot of signal over sample numbers
channels_to_display = ['AF1', 'CP3', 'F1']
for channel in channels_to_display:
    plt.xlabel('Sample number')
    plt.plot(signal_array[channel_list.index(channel)])
plt.legend(channels_to_display)

info_data = mne.create_info(ch_names=channel_list_fixed, sfreq=256, ch_types=['eeg']*64)
raw = mne.io.RawArray(signal_array, info_data)

standard_1020_montage = mne.channels.make_standard_montage('standard_1020')
raw.drop_channels(['X', 'Y', 'nd'])
raw.set_montage(standard_1020_montage)

raw.plot_psd()
raw.plot_psd(average=True)

raw_filtered = raw.copy().filter(8,30, verbose=False)
raw_filtered.plot_psd()
raw_filtered.plot_psd(average=True)

plt.imshow(raw.get_data())
plt.show()
plt.imshow(raw.copy().filter(1,10, verbose=False).get_data())
plt.show()
plt.plot(raw.copy().get_data()[40])
plt.plot(raw.copy().filter(8,30, verbose=False).get_data()[40])

ica = mne.preprocessing.ICA(random_state=42, n_components=20)
ica.fit(raw.copy().filter(1,None, verbose=False), verbose=False)
ica.plot_components()

plot_topomap(signal_array, show_names=False)


plot_joint_topomap(signal_array)




