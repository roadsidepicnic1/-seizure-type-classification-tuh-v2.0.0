import os
import sys
print('Current working path is %s' % str(os.getcwd()))
sys.path.insert(0, os.getcwd())

import platform
import argparse
import pandas as pd
import numpy as np
import math
import collections
from tabulate import tabulate
import pyedflib
import re
from scipy.signal import resample
import pickle
import h5py
import progressbar
from time import sleep

parameters = pd.read_csv('data_preparation/parameters.csv', index_col=['parameter'])
seizure_type_data = collections.namedtuple('seizure_type_data', ['patient_id','seizure_type', 'data'])

def generate_data_dict(xlsx_file_name,sheet_name, tuh_eeg_szr_ver):

    seizure_info = collections.namedtuple('seizure_info', ['patient_id','filename', 'start_time', 'end_time'])
    data_dict = collections.defaultdict(list)

    excel_file = os.path.join(xlsx_file_name)
    data = pd.read_excel(excel_file, sheet_name=sheet_name)
    if tuh_eeg_szr_ver == 'v1.5.2':
        data = data.iloc[1:]# remove first row
    elif tuh_eeg_szr_ver == 'v1.4.0':
        data = data.iloc[1:-4]  # remove first and last 4 rows
    else:
        exit('tuh_eeg_szr_ver %s is not supported'%tuh_eeg_szr_ver)

    col_l_file_name = data.columns[11]
    col_m_start = data.columns[12]
    col_n_stop = data.columns[13]
    col_o_szr_type = data.columns[14]
    train_files = data[[col_l_file_name, col_m_start,col_n_stop,col_o_szr_type]]
    train_files = np.array(train_files.dropna())

    for item in train_files:
        a = item[0].split('/')
        if tuh_eeg_szr_ver == 'v1.5.2':
            patient_id = a[4]
        elif tuh_eeg_szr_ver == 'v1.4.0':
            patient_id = a[5]
        else:
            exit('tuh_eeg_szr_ver %s is not supported' % tuh_eeg_szr_ver)

        v = seizure_info(patient_id = patient_id, filename = item[0], start_time=item[1], end_time=item[2])
        k = item[3] #szr_type
        data_dict[k].append(v)

    return data_dict

def print_type_information(data_dict):

    l = []
    for szr_type, szr_info_list in data_dict.items():
        # how many different patient id for seizure K?
        patient_id_list = [szr_info.patient_id for szr_info in szr_info_list]
        unique_patient_id_list,counts = np.unique(patient_id_list,return_counts=True)

        dur_list = [szr_info.end_time-szr_info.start_time for szr_info in szr_info_list]
        total_dur = sum(dur_list)
        # l.append([szr_type, str(len(szr_info_list)), str(len(unique_patient_id_list)), str(total_dur)])
        l.append([szr_type, (len(szr_info_list)), (len(unique_patient_id_list)), (total_dur)])

        #  numpy.asarray((unique, counts)).T
        '''
        if szr_type=='TNSZ':
            print('TNSZ Patient ID list:')
            print(np.asarray((unique_patient_id_list, counts)).T)
        if szr_type=='SPSZ':
            print('SPSZ Patient ID list:')
            print(np.asarray((unique_patient_id_list, counts)).T)
        '''

    sorted_by_szr_num = sorted(l, key=lambda tup: tup[1], reverse=True)
    print(tabulate(sorted_by_szr_num, headers=['Seizure Type', 'Seizure Num','Patient Num','Duration(Sec)']))

def merge_train_test(train_data_dict, dev_test_data_dict):

    merged_dict = collections.defaultdict(list)
    for item in train_data_dict:
        merged_dict[item] = train_data_dict[item] + dev_test_data_dict[item]

    return merged_dict

def extract_signal(f, signal_labels, electrode_name, start, stop):

    tuh_label = [s for s in signal_labels if 'EEG ' + electrode_name + '-' in s]

    if len(tuh_label) > 1:
        print(tuh_label)
        exit('Multiple electrodes found with the same string! Abort')

    channel = signal_labels.index(tuh_label[0])
    signal = np.array(f.readSignal(channel))

    start, stop = float(start), float(stop)
    original_sample_frequency = f.getSampleFrequency(channel)
    original_start_index = int(np.floor(start * float(original_sample_frequency)))
    original_stop_index = int(np.floor(stop * float(original_sample_frequency)))

    seizure_signal = signal[original_start_index:original_stop_index]

    new_sample_frequency = int(parameters.loc['sampling_frequency']['value'])
    new_num_time_points = int(np.floor((stop - start) * new_sample_frequency))
    seizure_signal_resampled = resample(seizure_signal, new_num_time_points)

    return seizure_signal_resampled

def read_edfs_and_extract(edf_path, edf_start, edf_stop):

    f = pyedflib.EdfReader(edf_path)

    montage = str(parameters.loc['montage']['value'])
    montage_list = re.split(';', montage)
    signal_labels = f.getSignalLabels()
    x_data = []

    for i in montage_list:
        electrode_list = re.split('-', i)
        electrode_1 = electrode_list[0]
        extracted_signal_from_electrode_1 = extract_signal(f, signal_labels, electrode_name=electrode_1, start=edf_start, stop=edf_stop)
        electrode_2 = electrode_list[1]
        extracted_signal_from_electrode_2 = extract_signal(f, signal_labels, electrode_name=electrode_2, start=edf_start, stop=edf_stop)
        this_differential_output = extracted_signal_from_electrode_1-extracted_signal_from_electrode_2
        x_data.append(this_differential_output)

    f._close()
    del f

    x_data = np.array(x_data)

    return x_data

def load_edf_extract_seizures_v200(base_dir, save_data_dir, data_dict):
    for patient in data_dict:
        edf_file = data_dict[patient]['edf']
        save_patient_dir = os.path.join(save_data_dir, patient)
        os.makedirs(save_patient_dir, exist_ok=True)
        save_data_file = os.path.join(save_patient_dir, patient + '.h5')
        with h5py.File(save_data_file, 'w') as hf:
            for seizure_data in data_dict[patient]['seizure_data']:
                seizure_type = seizure_data.seizure_type
                start = seizure_data.data[0]
                stop = seizure_data.data[1]
                with pyedflib.EdfReader(edf_file) as f:
                    signal_labels = f.getSignalLabels()
                    data = extract_signal(f, signal_labels, 'EEG Fpz-Cz', start, stop)
                    hf.create_dataset(seizure_type, data=data)


def main():
    base_dir = 'D:\ v2.0.0'
    save_data_dir = 'D:\seizureergebnisse'
    tuh_eeg_szr_ver = '2.0.0'
    data_dict = generate_data_dict(base_dir, tuh_eeg_szr_ver)
    print_type_information(data_dict)
    load_edf_extract_seizures_v200(base_dir, save_data_dir, data_dict)
    merge_train_test(data_dict['train'], data_dict['dev_test'])
    ...
    # Weiterer Code

if __name__ == '__main__':
    main()