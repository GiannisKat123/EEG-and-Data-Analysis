import pandas as pd
import numpy as np
import os
from typing import Dict
from scipy import stats
from mne_features.univariate import *
from mne_features.bivariate import *
import pywt
from statsmodels.regression.linear_model import yule_walker,burg
from statsmodels.tsa.stattools import adfuller
from scipy.stats import pearsonr
import antropy as ant
from scipy.signal import hilbert
from sklearn.preprocessing import MinMaxScaler


def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simps
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf
    
    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)
    # print(bp,band)
    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp


class Feature_Extractor():
    def __init__(self,data:pd.DataFrame|np.ndarray,channels:Dict[str,int]):
        self.channels = channels
        self.channels_keys = list(channels.keys())
        self.columns = ['channel']
        self.data = data if type(data) == np.ndarray else data.to_numpy()
        self.list_of_features = [[_] for _ in self.channels_keys]
        self.channels_to_id = {chan:index for index,chan in enumerate(self.channels_keys)}

    def check_channels_in_func(self,channels:Dict[str,int]):
        for chan in list(channels.keys()):
            if (chan in self.channels_keys and channels[chan] == self.channels[chan]) == False:return False
        return True
    
    def create_new_directory(self,path:str,name:str):
        new_directory = path + f"\\{name}"
        if os.path.exists(new_directory) == False:
            os.mkdir(new_directory)
        return new_directory

    def calc_stat(self,channels_dict:Dict[str,int],column_name:str,stat_func):
        if self.check_channels_in_func(channels_dict):
            self.columns.append(column_name)
            for chan in list(channels_dict.keys()):
                if chan in self.channels_keys and channels_dict[chan] == self.channels[chan]:
                    channel_values = self.data[:,self.channels[chan]]
                    self.list_of_features[self.channels_to_id[chan]].append(stat_func(channel_values))

    def calc_mean(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'mean_value',np.mean)

    def calc_median(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'median_value',np.median)

    def calc_standard_deviation(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'std_value',np.std)

    def calc_peak_to_peak(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'ptp_value',np.ptp)

    def calc_min_value(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'min_value',np.min)

    def calc_argmin_value(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'argmin_value',np.argmin)

    def calc_max_value(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'max_value',np.max)

    def calc_argmax_value(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'argmax_value',np.argmax)

    def calc_25th_percentile_value(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'25th_percentile_value',lambda x: np.percentile(x,25))

    def calc_75th_percentile_value(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'75th_percentile_value',lambda x:np.percentile(x,75))

    def calc_skew_value(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'skew_value',stats.skew)

    def calc_kurtosis_value(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'kurtosis_value',stats.kurtosis)

    def calc_rms_value(self,channels_dict:Dict[str,int]):
        self.calc_stat(channels_dict,'rms_value',lambda x: np.sqrt(np.mean(x**2)))

    def hjorth_parameters(self,channels_dict:Dict[str,int]):
        if self.check_channels_in_func(channels_dict):
            self.columns += ['Hjorth_activity','Hjorth_mobility','Hjorth_complexity']
            for chan in list(channels_dict.keys()):
                if chan in self.channels_keys and channels_dict[chan] == self.channels[chan]:
                    channel_values = self.data[:,self.channels[chan]]
                    activity = np.var(channel_values)
                    mobility = compute_hjorth_mobility(channel_values)
                    complexity = compute_hjorth_complexity(channel_values)
                    self.list_of_features[self.channels_to_id[chan]] += [activity,mobility,complexity]
    
    def bandpower_in_freq_bands(self,channels_dict:Dict[str,int],freq_bands:Dict[str,list[int,int]]):
        if self.check_channels_in_func(channels_dict):
            self.columns += [f'bandpower_{band}' for band in list(freq_bands.keys())]
            for chan in list(channels_dict.keys()):
                if chan in self.channels_keys and channels_dict[chan] == self.channels[chan]:
                    channel_values = self.data[:,self.channels[chan]]
                    avg_power = compute_pow_freq_bands(sfreq=256,data=np.array([channel_values]),freq_bands=freq_bands,normalize=False)
                    features_per_channel = [avg_power[i] for i in range(len(avg_power))]
                    self.list_of_features[self.channels_to_id[chan]] += features_per_channel
    
    def wavelet_decomposition(self,channels_dict:Dict[str,int],type_of_wavelet:str,order:int,features_extracted:list[str]|None):
        if self.check_channels_in_func(channels_dict):
            wavelet = pywt.Wavelet(type_of_wavelet)
            for chan in list(channels_dict.keys()):
                if chan in self.channels_keys and channels_dict[chan] == self.channels[chan]:
                    channel_values = self.data[:,self.channels[chan]]
                    max_level = pywt.dwt_max_level(len(channel_values),wavelet.dec_len)
                    max_val = min(max_level,order)
                    coefs = pywt.wavedec(channel_values,wavelet=type_of_wavelet,level=max_val,mode='symmetric')
                    if not features_extracted:
                        return coefs
                    for i in range(len(coefs)):
                        for feat in features_extracted:
                            if 'mean' == feat: self.list_of_features[self.channels_to_id[chan]].append(np.mean(coefs[i]))
                            if 'std'  == feat: self.list_of_features[self.channels_to_id[chan]].append(np.std(coefs[i]))
                            if 'min'  == feat: self.list_of_features[self.channels_to_id[chan]].append(np.min(coefs[i]))
                            if 'max'  == feat: self.list_of_features[self.channels_to_id[chan]].append(np.max(coefs[i]))
                            if 'average_power'  == feat: self.list_of_features[self.channels_to_id[chan]].append(np.mean(np.abs(coefs[i] ** 2)))
                            if 'sum_power' == feat: self.list_of_features[self.channels_to_id[chan]].append(np.sum(coefs[i]**2))
            
            for i in range(max_val+1):
                for feat in features_extracted:
                    self.columns.append(f'wavelet_coef_{i+1}_{feat}')

    def ar_coefficients(self,channels_dict:Dict[str,int],method:str,order:int|None):
        if self.check_channels_in_func(channels_dict):
            if not order:
                list_of_orders = []
                for chan in list(channels_dict.keys()):
                    if chan in self.channels_keys and channels_dict[chan] == self.channels[chan]:
                        channel_values = self.data[:,self.channels[chan]] * (10**6)
                        y = np.diff(channel_values)
                        df_stationarityTest = adfuller(y,autolag='AIC')
                        list_of_orders.append(df_stationarityTest[2])
                order = int(np.mean(list_of_orders) + 0.5)
            
            self.columns += [f'ar_coef_{i+1}_{method}' for i in range(order)] 
            for chan in list(channels_dict.keys()):
                if chan in self.channels_keys and channels_dict[chan] == self.channels[chan]:
                    channel_values = self.data[:,self.channels[chan]] * (10**6)
                    y = np.diff(channel_values)
                    if method == 'yule_walker': coeffs,sigma = yule_walker(y,order)               
                    elif method == 'burg': coeffs,sigma = burg(y,order)
                    else: return None
                    self.list_of_features[self.channels_to_id[chan]] += [coef for coef in coeffs]
                    
    def entropy(self,channels_dict:Dict[str,int],entropy_measures:list[str]):
        if self.check_channels_in_func(channels_dict):
            self.columns += entropy_measures
            for chan in list(channels_dict.keys()):
                if chan in self.channels_keys and channels_dict[chan] == self.channels[chan]:
                    channel_values = self.data[:,self.channels[chan]]
                    if 'sample' in entropy_measures: self.list_of_features[self.channels_to_id[chan]].append(ant.sample_entropy(channel_values)) 
                    if 'permutation' in entropy_measures: self.list_of_features[self.channels_to_id[chan]].append(ant.perm_entropy(channel_values,normalize=True)) 
                    if 'approximate' in entropy_measures: self.list_of_features[self.channels_to_id[chan]].append(ant.app_entropy(channel_values)) 
                    if 'spectral' in entropy_measures: self.list_of_features[self.channels_to_id[chan]].append(ant.spectral_entropy(channel_values,sf=256,method='welch',normalize=True)) 

    def pearson_corr(self,channels_dict:Dict[str,int]):
        if self.check_channels_in_func(channels_dict):
            for chan1 in list(channels_dict.keys()):
                if chan1 in self.channels_keys and channels_dict[chan1] == self.channels[chan1]:
                    channel_values1 = scale(self.data[:,self.channels[chan1]],axis=0)
                    self.columns.append(f'pearson_corr_{chan1}')
                    for chan2 in list(channels_dict.keys()):
                        if chan2 in self.channels_keys and channels_dict[chan2] == self.channels[chan2]:
                            channel_values2 = scale(self.data[:,self.channels[chan2]],axis=0)
                            self.list_of_features[self.channels_to_id[chan1]].append(pearsonr(channel_values1,channel_values2)[0])

    def spectrum_corr(self,channels_dict:Dict[str,int]):
        if self.check_channels_in_func(channels_dict):
            for chan1 in list(channels_dict.keys()):
                if chan1 in self.channels_keys and channels_dict[chan1] == self.channels[chan1]:
                    channel_values1,_ = power_spectrum(sfreq=256,data=self.data[:,self.channels[chan1]],psd_method='welch')
                    self.columns.append(f'spectrum_corr_{chan1}')
                    for chan2 in list(channels_dict.keys()):
                        if chan2 in self.channels_keys and channels_dict[chan2] == self.channels[chan2]:
                            channel_values2,_ = power_spectrum(sfreq=256,data=self.data[:,self.channels[chan2]],psd_method='welch')           
                            self.list_of_features[self.channels_to_id[chan1]].append(pearsonr(channel_values1,channel_values2)[0])

    def PLV(self,channels_dict:Dict[str,int]):
        if self.check_channels_in_func(channels_dict):
            for chan1 in list(channels_dict.keys()):
                if chan1 in self.channels_keys and channels_dict[chan1] == self.channels[chan1]:
                    analytical_signal1 = hilbert(self.data[:,self.channels[chan1]])
                    phase1 = np.angle(analytical_signal1)
                    self.columns.append(f'PLV_{chan1}')
                    for chan2 in list(channels_dict.keys()):
                        if chan2 in self.channels_keys and channels_dict[chan2] == self.channels[chan2]:
                            analytical_signal2 = hilbert(self.data[:,self.channels[chan2]])
                            phase2 = np.angle(analytical_signal2)
                            phase_diff = phase1 - phase2
                            self.list_of_features[self.channels_to_id[chan1]].append(np.abs(np.mean(np.exp(1j*phase_diff))))

    def PLI(self,channels_dict:Dict[str,int]):
        if self.check_channels_in_func(channels_dict):
            for chan1 in list(channels_dict.keys()):
                if chan1 in self.channels_keys and channels_dict[chan1] == self.channels[chan1]:
                    analytical_signal1 = hilbert(self.data[:,self.channels[chan1]])
                    phase1 = np.angle(analytical_signal1)
                    self.columns.append(f'PLI_{chan1}')
                    for chan2 in list(channels_dict.keys()):
                        if chan2 in self.channels_keys and channels_dict[chan2] == self.channels[chan2]:
                            analytical_signal2 = hilbert(self.data[:,self.channels[chan2]])
                            phase2 = np.angle(analytical_signal2)
                            phase_diff = phase1 - phase2
                            self.list_of_features[self.channels_to_id[chan1]].append(np.abs(np.mean(np.sign(np.exp(1j*phase_diff)))))

    def empty_file(self):
        self.columns = []
        self.list_of_features = [[_] for _ in self.channels_keys]

    def save_features(self,directory:str,filename:str):
        df = pd.DataFrame(self.list_of_features,columns = self.columns)
        print(df)
        df.fillna(value=-1.0, inplace=True)  # Replaces NaN with 0
        print(df)
        df.to_csv(directory+f"//{filename}.csv")

    def unionize_files(self,directory:str,list_of_files:list[str],normalization:bool):
        list_of_csvs = []
        columns = []
        channels = []
        file_names = []

        list_of_files_csv = []
        for file in list_of_files:
            if '.csv' in file:
                list_of_files_csv.append(file)

        for file in list_of_files_csv:
            filename = file.split('.csv')[0]
            file_names.append(filename)
            df = pd.read_csv(directory + "\\" + file)
            list_of_csvs.append(df)
            df_columns = list(df.columns)[2:]
            columns += [col for col in df_columns if col not in columns]
            channels += [chan for chan in df['channel'] if chan not in channels]
        
        # matrix_form_cols = ['channel'] + columns
        # for index,df in enumerate(list_of_csvs):
        #     data_df = []
        #     for chan in range(len(channels)):
        #         data = [channels[chan]] + [df[col][chan] if (col in df.columns and df['channel'][chan] == channels[chan]) else -1.0 for col in columns]
        #         data_df.append(data)
        #     df = pd.DataFrame(data_df,columns=matrix_form_cols)
        #     if os.path.exists(directory + '\\newform') == False:
        #         os.mkdir(directory + '\\newform')
        #     df.to_csv(directory + '\\newform'+f'\\{file_names[index]}.csv')

        deleted_columns = [f'pearson_corr_{chan}_{chan}' for chan in channels] + [f'PLI_{chan}_{chan}' for chan in channels] + [f'PLV_{chan}_{chan}' for chan in channels]

        filtered_cols = [f'{col}_{chan}' for col in columns for chan in channels if (f'{col}_{chan}' not in deleted_columns)]

        new_columns = [f'{col}_{chan}' for col in columns for chan in channels]

        data_df = [[] for i in range(len(list_of_files_csv))]

        for index,df in enumerate(list_of_csvs):
            data_df[index] += [df[col][chan] if (f'{col}_{channels[chan]}' in filtered_cols and col in df.columns and df['channel'][chan] == channels[chan]) else -1.0 for col in columns for chan in range(len(channels))]
        df = pd.DataFrame(data_df,columns=new_columns)
        
        print(df)


        for col in df.columns:
            if (df[col] == df[col].iloc[0]).all():
                df.drop(col, axis=1, inplace=True)
        
        print(df)

        if normalization:
            scaler = MinMaxScaler()
            df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            df = df_normalized
        
        print(df)

        if os.path.exists(directory + '\\unionized_file') == False:
            os.mkdir(directory + '\\unionized_file')
        df.to_csv(directory + '\\unionized_file'+'\\unionized_file.csv')

