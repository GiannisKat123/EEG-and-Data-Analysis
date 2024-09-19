from EEG_feature_extraction import Feature_Extractor
import os
import pandas as pd

real_path = os.path.realpath(__file__)
dir_path = os.path.dirname(real_path).split('\\')
path = ""
for i in range(len(dir_path)):
    path = path + dir_path[i] + "\\"

print(path)

users = [f"user{i}" for i in range(1,2)]
epoch_types = ["start_trial", "gesture1", "gesture2", "gesture3", "full_trial"]

for user in users:
    path_user = path + user + "\\"
    dir_path = path_user + f'MNE_pipeline_without_muscleartifacts_{user}'
    dir_list = os.listdir(dir_path)

    ### Read files from the folder and create a folder for the created features with suitable names (from epochs created in the preprocessing phase)
    for file in dir_list:
        df = pd.read_csv(dir_path + "\\" + file)
        df = df.iloc[:,2:]
        channels = list(df.columns)
        channels_dict = {channels[i]: i for i in range(len(channels))}
        fe = Feature_Extractor(df,channels_dict)
        
        ### Creation of folder for feature files\
        trial_num = file.split('_')[1]
        file_name = file.split(".csv")[0]
        epoch_name = ""
        familiarity = ""
        for epoch in epoch_types:
            if epoch in file_name:
                epoch_name = epoch
                break 
        if epoch_name == "": continue
        if 'unfamiliar' in file_name:familiarity = 'unfamiliar'
        else: familiarity = 'familiar'

        dir_file = fe.create_new_directory(path_user,f'features_{familiarity}_{epoch_name}')

        print(dir_file,f'epoch_{trial_num}')
        
        fe.calc_mean(channels_dict)
        fe.calc_median(channels_dict)
        fe.calc_standard_deviation(channels_dict)
        fe.calc_peak_to_peak(channels_dict)
        fe.calc_min_value(channels_dict)
        fe.calc_argmin_value(channels_dict)
        fe.calc_max_value(channels_dict)
        fe.calc_argmax_value(channels_dict)
        fe.calc_25th_percentile_value(channels_dict)
        fe.calc_75th_percentile_value(channels_dict)
        fe.calc_skew_value(channels_dict)
        fe.calc_kurtosis_value(channels_dict)
        fe.calc_rms_value(channels_dict)
        fe.hjorth_parameters(channels_dict)
        
        freq_bands = {"delta": [1, 4],"theta": [4, 8],"alpha": [8, 13],"beta":[13, 30],"gamma": [30, 100]}
        fe.bandpower_in_freq_bands(channels_dict,freq_bands)

        fe.wavelet_decomposition(channels_dict,type_of_wavelet='db4',order=5,features_extracted=['mean','std','min','max','average_power','sum_power'])
        fe.ar_coefficients(channels_dict,method='yule_walker',order=6)
        fe.ar_coefficients(channels_dict,method='burg',order=6)
        fe.entropy(channels_dict,entropy_measures=['sample','permutation','approximate','spectral'])
        fe.pearson_corr(channels_dict)
        fe.PLV(channels_dict)
        fe.PLI(channels_dict)
        fe.save_features(dir_file,f'epoch_{trial_num}')

        # print(dir_file)
        # files = os.listdir(dir_file)
        # fe.unionize_files(dir_file,files,True)





