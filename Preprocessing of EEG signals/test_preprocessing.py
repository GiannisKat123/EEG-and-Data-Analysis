import os
import pandas as pd
from EEG_preprocessing import EEG_data_preprocessing
import numpy as np

EEG_data_processor = EEG_data_preprocessing()

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path).split('\\')
path = ""
for i in range(len(dir_path)):
    path = path + dir_path[i] + "\\"

users = [f"user{i}" for i in range(1,4)]

for user in users:
    path_user = path + user
    new_directory = EEG_data_processor.create_new_folder(path_user,f"MNE_pipeline_with_muscle_{user}")

    df = pd.read_csv(path_user + "//{}_on_user1_PGA_new.csv".format(user))
    
    events = []
    dir_list = os.listdir(path_user + "//trials_raw_phase2")

    for i in range(len(dir_list)):
        sample_data_raw_file = path_user + "//trials_raw_phase2" + f"//trial_{i+1}_raw.fif"
        signal_data = EEG_data_processor.load_file(file_path = sample_data_raw_file)
        
        ### Channels that are not used in my case
        channels = signal_data['channels']
        cleared_signal = EEG_data_processor.drop_bad_channels_by_hand(signal_data['signal'],channels[32:])
        new_channels = channels[:32]

        # Creation of STIM channel for events created by my application
        sampling_frequency = signal_data['sampling_freq']
        experiment_length = signal_data['data_length']
        df_help = cleared_signal.to_data_frame()

        familiarity = "familiar"
        if df["fam/unfam"][i] == 0: 
            familiarity = "unfamiliar"

        timestamps = []
        times = df_help['time']
        event_times = [df["start_time"][i],df["gesture1_time1"][i],df["gesture1_time2"][i],df["gesture2_time1"][i],df["gesture2_time2"][i],df["gesture3_time1"][i],df["gesture3_time2"][i],df["end_time"][i]]
        for event in event_times:
            sample = int(event * sampling_frequency)
            min_dif = (abs(times[sample] - event),sample) #time,index
            for j in range(sample-10,sample):
                if abs(times[j] - times[sample]) < min_dif[0]: 
                    min_dif = (abs(times[j] - times[sample]),j)
            for j in range(sample+1,sample+11):
                if abs(times[j] - times[sample]) < min_dif[0]: 
                    min_dif = (abs(times[j] - times[sample]),j)
            timestamps.append([times[min_dif[1]],min_dif[1],familiarity]) 

        L = [0 for i in range(len(df_help))]
        for j in range(len(timestamps)):
            L[timestamps[j][1]] = 1000000.0
        
        df_help["STIM"] = L 
        stim_data = df_help["STIM"]

        ######## Insert STIM channel to data

        new_signal,events = EEG_data_processor.create_events(cleared_signal,stim_data,"STIM")
        
        for x in range(len(events)):
            events[x][2] = x+1

        event_dict = {
            "start_trial": 1,
            "gesture1_start": 2,
            "gesture1_end": 3,
            "gesture2_start": 4,
            "gesture2_end": 5,
            "gesture3_start": 6,
            "gesture3_end": 7,
            "end_trial": 8
        }

        #### Load biosemi montage
        biosemi_montage = EEG_data_processor.load_montage("biosemi32")
        
        ### 1) MNE pipeline
        
        # ## Import biosemi montage
        # mne_signal = EEG_data_processor.implement_montage(new_signal,biosemi_montage)
        
        # ## Rereferencing of EEG signals
        # rereferenced_signal,rereferenced_data = EEG_data_processor.rereference_eeg_channels(eeg_signal = mne_signal,rereferenced_channels='average')
        
        # ## Notch filtering at 50Hz
        # preprocessed_signal = EEG_data_processor.filtering_data(signal = rereferenced_signal,notch_freq=50.0,method='spectrum_fit')
        
        ### 2) PREP Pipeline
        preprocessed_signal = EEG_data_processor.prep_pipeline_preprocessing(montage = biosemi_montage,ref_chs='eeg',reref_chs='eeg',line_freq=50.0,signal=new_signal)

        ### Bandpass filter between 1-100 Hz
        filtered_signal = EEG_data_processor.filtering_data(signal=preprocessed_signal,l_freq = 1.0,h_freq = 100.0, method = 'fir')
        
        ### Resampling of EEG data to 256Hz
        resampled_freq = 256.0
        resampled_signal,resampled_events = EEG_data_processor.resampling(signal=filtered_signal,sampling_freq=resampled_freq,events_=events)

        ### Artifact Rejection
        cleared_signal = EEG_data_processor.ica_repairing_artifacts(signal=resampled_signal,ica_n_components=0.99,events=resampled_events,excluded_artifacts=["heart beat","eye blink",'line noise'])
       
        ### Creation of epochs
        categories = ["before_gestures","start_trial", "gesture1", "gesture2", "gesture3", "full_trial"]
        id_event = [1,1,2,4,6,1]

        times_max = [0.5,abs(resampled_events[0][0]/resampled_freq-resampled_events[1][0]/resampled_freq),abs(resampled_events[1][0]/resampled_freq-resampled_events[2][0]/resampled_freq),
                     abs(resampled_events[3][0]/resampled_freq-resampled_events[4][0]/resampled_freq),abs(resampled_events[5][0]/resampled_freq-resampled_events[6][0]/resampled_freq),abs(resampled_events[0][0]/resampled_freq-resampled_events[-1][0]/resampled_freq)]
        
        new_events = []
        
        for id in id_event:
            new_events.append([resampled_events[id][0],resampled_events[id][1],resampled_events[id][2]])
        
        new_events = np.array(new_events)        
        
        tmin = -0.5
        baseline = (None, 0)
        reject_criteria = dict(eeg=150e-6)
        preload = True

        for x in range(len(categories)):
            tmax = times_max[x] + 0.5
            category = categories[x]
            event = list(event_dict.keys())[id_event[x]-1]
            event_d = {
                f"{event}": id_event[x]
            }
            epochs = EEG_data_processor.creation_of_epochs(cleared_signal,resampled_events,event_d,tmin,tmax,baseline,reject_criteria,preload)
            EEG_data_processor.save_epochs_to_directory(path = new_directory, name = f'epoch_{i+1}_PREP_{familiarity}_{category}', epochs = epochs, channels = new_channels, tmin = tmin, tmax = tmax)    













