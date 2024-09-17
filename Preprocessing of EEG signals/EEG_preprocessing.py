import mne
from pyprep.prep_pipeline import PrepPipeline
from pyprep.find_noisy_channels import NoisyChannels
from mne_icalabel import label_components
from autoreject import get_rejection_threshold
import os
import pandas as pd
import numpy as np

class EEG_data_preprocessing:
    def __init__(self):
        self.file_path:str = None
        self.raw:mne.io.Raw = None
        self.sampling_frequency:float = None
        self.experiment_length:float = None
        self.experiment_samples: int = None
        self.channels: str | list[str] = None
        self.datetime_start: str = None

    def load_file(self,file_path:str):
        #### Load data
        self.file_path = file_path
        self.raw = mne.io.read_raw_fif(self.file_path)
        self.raw.load_data()
        self.get_info()
        return {'signal':self.raw,
                'sampling_freq':self.sampling_frequency,
                'data_length':self.experiment_length,
                'data_samples':self.experiment_samples,
                'channels':self.channels,
                'start_time':self.datetime_start}
        
    def get_info(self):
        print(self.raw.info)
        self.sampling_frequency = self.raw.info['sfreq']
        self.experiment_length = len(self.raw.times)/self.sampling_frequency
        self.experiment_samples = len(self.raw.times)
        self.channels = self.raw.info['ch_names']
        self.datetime_start = self.raw.info['meas_date']
        print(type(self.datetime_start),self.datetime_start)
    
    def drop_bad_channels_by_hand(self,signal:mne.io.Raw,bad_channel:str|list[str]):
        signal.drop_channels(bad_channel)
        return signal
    
    def rereference_eeg_channels(self,eeg_signal:mne.io.Raw,rereferenced_channels:list[str]):
        return mne.set_eeg_reference(eeg_signal,ref_channels=rereferenced_channels,copy=True)
    
    def create_events(self,signal:mne.io.Raw,stim_channel:pd.Series,stim_channel_name:str):
        info = mne.create_info(['STIM'],self.sampling_frequency,['stim'])
        stim_data = [stim_channel.to_numpy()]
        stim_raw = mne.io.RawArray(stim_data,info)
        print(stim_raw)
        signal_new = signal.add_channels([stim_raw],force_update_info=True)
        events = mne.find_events(signal_new,stim_channel='STIM')
        signal_new.drop_channels(['STIM'])
        return signal_new,events
        
    def load_montage(self,montage_name:str):
        montage = mne.channels.make_standard_montage(f'{montage_name}')   
        if montage_name == "biosemi32":
            missing_channels =  ['Af3', 'Fc1', 'Fc5', 'Cp1', 'Cp5', 'Po3', 'Po4', 'Cp6', 'Cp2', 'Fc6', 'Fc2', 'Af4']
            for x in range(len(missing_channels)):
                for j in range(len(montage.ch_names)):
                    if len(montage.ch_names[j]) == 3 and montage.ch_names[j].lower().capitalize() == missing_channels[x]:
                        montage.ch_names[j] = missing_channels[x]
        return montage
    
    def implement_montage(self,eeg_signal:mne.io.Raw,montage:mne.channels.DigMontage):
        eeg_signal.set_montage(montage)
        return eeg_signal
    
    def prep_pipeline_preprocessing(self,montage:mne.channels.DigMontage,ref_chs:str,reref_chs:str,line_freq:float,signal:mne.io.Raw):
        
        line_freqs = np.arange(line_freq, self.sampling_frequency / 2, line_freq)

        prep_params = {
            'ref_chs': ref_chs,
            'reref_chs': reref_chs,
            'line_freqs':line_freqs
        }
        
        signal_copy = signal.copy()
        prep = PrepPipeline(signal_copy,prep_params=prep_params,montage=montage)
        prep.fit()
                
        signal_preprocessed = prep.raw_eeg
        
        #### Interpolating the bad channels one more time
        try:
            signal_temp = signal_preprocessed.copy().interpolate_bads(reset_bads=True)
            return signal_temp
        except RuntimeWarning:
            return signal_preprocessed
            
    def filtering_data(self, signal:mne.io.Raw,l_freq:None|float = None,h_freq:None|float = None,notch_freq:None|float = None, method:str = 'fir'):
        
        if (l_freq,h_freq) == (None,None) and notch_freq and self.sampling_frequency:
            filtered_signal = signal.notch_filter(
                freqs = np.arange(notch_freq, self.sampling_frequency / 2, notch_freq),
                method = method
            )
            return filtered_signal
        
        elif notch_freq == None and (l_freq,h_freq) != (None,None):
            filtered_signal = signal.filter(l_freq=l_freq,h_freq=h_freq,method=method)        
            return filtered_signal
        
        else:
            return "Something is wrong buddy"
      
    def resampling(self,signal:mne.io.Raw,sampling_freq:float,events_:list[list]|list):
        resampled_signal,resampled_events = signal.copy().resample(sfreq=sampling_freq,events=events_)
        self.sampling_frequency = sampling_freq
        self.experiment_length = int(len(resampled_signal.times) / sampling_freq)
        return resampled_signal,resampled_events
    
    def ica_repairing_artifacts(self,signal:mne.io.Raw,ica_n_components:float|int,events:list[list]|list,excluded_artifacts:list[str]|None):
        random_state = 42
        ica = mne.preprocessing.ICA(n_components=ica_n_components,max_iter="auto",random_state=random_state)
        ica.fit(signal)
        
        ic_labels = label_components(signal,ica,method='iclabel')

        labels = ic_labels["labels"]
        print(labels)

        exclude_idx = None

        if excluded_artifacts:
            exclude_idx = [idx for idx,label in enumerate(labels) if label in excluded_artifacts]
        # exclude_idx = [idx for idx,label in enumerate(labels) if label not in ["brain","other"]]

        reconstructed_signal = signal.copy()
        reconstructed_signal_1 = ica.apply(reconstructed_signal,exclude=exclude_idx)

        return reconstructed_signal_1
    
    def creation_of_epochs(self,signal:mne.io.Raw,events_:list[list]|list,event_dict:dict,tmin_:float,tmax_:float,baseline_:tuple,reject_criteria_:dict,preload_:bool):
        epochs = mne.Epochs(raw=signal,events=events_,event_id=event_dict,tmin=tmin_,tmax=tmax_,baseline=baseline_,reject=reject_criteria_,preload=preload_)
        return epochs
    
    def save_epochs_to_directory(self,path:str,name:str,epochs:mne.epochs.Epochs,channels:list[str],tmin:float,tmax=float):
        if len(epochs)!=0:
            data = []
            headers = ["time"] + channels
            
            df_epochs = epochs.to_data_frame(index=["condition", "epoch", "time"])
            times = np.linspace(start=tmin,stop=tmax,num=len(df_epochs))
            data_epochs = epochs.get_data()
                        
            for j in range(len(times)):
                data_per_time = [times[j]]
                for channel_num in range(len(channels)):
                    data_per_time.append(data_epochs[0][channel_num][j])
                data.append(data_per_time)
            
            df_new = pd.DataFrame(data,columns=headers)
            df_new.to_csv(path + f'//{name}' + '.csv')
        else:
            print("No epochs to create file for")


    def plot_data(self,signal:mne.io.Raw):
        signal.plot()
        signal.plot_psd()

    def create_new_folder(self,path_name,name):
        new_directory = path_name + f"//{name}"
        if not os.path.exists(new_directory):
            os.mkdir(new_directory)
        return new_directory
