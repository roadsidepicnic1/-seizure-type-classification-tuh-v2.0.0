import csv
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


def generate_data_dict_v200(base_dir, sheet_name, tuh_eeg_szr_ver):

    eeg_files = os.path.join(base_dir, 'v2.0.0' , 'edf' , sheet_name)  #D:\v2.0.0\eeg\train

    seizure_info = collections.namedtuple('seizure_info', ['patient_id', 'filename', 'start_time',
                                                    'end_time'])  # named tuple, allows access to fields through name
    data_dict = collections.defaultdict(list)  # dict, standardvalue for keys, standardvalue empty list(list)

    patient_id = 0


    for patient in os.listdir(eeg_files):
        patient_id = patient_id+1
        patient_path = os.path.join(eeg_files, patient)
        for session in os.listdir(patient_path):
            session_path = os.path.join(patient_path, session)
            for file_name in os.listdir(session_path):
                file_path = os.path.join(session_path, file_name)
                for seizure_data in os.listdir(file_path):
                    seizure_data_path = os.path.join(file_path, seizure_data)
                    if seizure_data.endswith('.csv_bi'):    #eventbased
                        file_name_dic = os.path.join(sheet_name, patient, session, file_name, seizure_data) #for dict
                        with open(seizure_data_path, 'r') as term_based_data:
                            reader = csv.reader(term_based_data)
                            for _ in range(6):
                                next(reader)
                            for row in reader:
                                start = float(row[1])
                                end = float(row[2])
                                seiz_typ = row[3]
                                #row[0]= term, row[1]=start, row[2]= end, row[3]=type(seiz,bckg) row[4] = confidence
                                if (row[3] == "seiz"):
                                    path_to_csv = os.path.join(base_dir, 'v2.0.0', 'edf' ,sheet_name, patient, session, file_name)
                                    for dir in os.listdir(path_to_csv):
                                        if dir.endswith('.csv'):
                                            csv_file = os.path.join(path_to_csv, dir)
                                            with open(csv_file, "r") as event_based_data:
                                                reader_event_based = csv.reader(event_based_data)
                                                for _ in range(6):
                                                    next(reader_event_based)
                                                for rowx in reader_event_based :
                                                    if float(rowx[1]) == float(start) and float(rowx[2]) == float(end) :
                                                        seiz_typ = rowx[3]

                            if seiz_typ == "seiz":
                                seiz_typ = "gnsz"
                            v = seizure_info(patient_id = str(patient_id), filename=file_name_dic, start_time= float(row[1]), end_time=float(row[2]))
                            k = seiz_typ
                            data_dict[k].append(v)

                    """   #termbased
                    if seizure_data.endswith('.csv'):
                        with open(seizure_data_path, 'r') as term_based_data:
                            reader = csv.reader(term_based_data)
                            for row in reader:
                                print(row)
                    """
    """
    for seizure_type, seizure_info_list in data_dict.items():
        print("Seizure Type:", seizure_type)
        for seizure_info in seizure_info_list:
            print("Patient ID:", seizure_info.patient_id)
            print("Filename:", seizure_info.filename)
            print("Start Time:", seizure_info.start_time)
            print("End Time:", seizure_info.end_time)
            print("-------------------")
        break
    """

    return data_dict

def generate_data_dict(xlsx_file_name,sheet_name, tuh_eeg_szr_ver):

    seizure_info = collections.namedtuple('seizure_info', ['patient_id','filename', 'start_time', 'end_time'])
    data_dict = collections.defaultdict(list)
    excel_file = os.path.join(xlsx_file_name)
    data = pd.read_excel(excel_file, sheet_name=sheet_name)
    if tuh_eeg_szr_ver == 'v1.5.2':
        data = data.iloc[1:]# remove first row
    elif tuh_eeg_szr_ver == 'v1.4.0':
        data = data.iloc[1:-4]  # remove first and last 4 rows
    elif tuh_eeg_szr_ver == 'v2.0.0':
        data = data.iloc[1:]
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
        elif tuh_eeg_szr_ver == 'v2.0.0':
            patient_id= a[6]
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

def load_edf_extract_seizures_v140(base_dir, save_data_dir, data_dict):

    seizure_data_dict = collections.defaultdict(list)

    count = 0
    bar = progressbar.ProgressBar(maxval=sum(len(v) for k, v in data_dict.items()),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for seizure_type, seizures in data_dict.items():
        for seizure in seizures:
            rel_file_location = seizure.filename.replace('.tse', '.edf').replace('./', '')
            patient_id = seizure.patient_id
            abs_file_location = os.path.join(base_dir,rel_file_location)
            temp = seizure_type_data(patient_id = patient_id, seizure_type = seizure_type, data = read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time))
            with open(os.path.join(save_data_dir, 'szr_' + str(count) + '_pid_' + patient_id + '_type_' + seizure_type + '.pkl'), 'wb') as fseiz:
                pickle.dump(temp, fseiz)
            count += 1
            bar.update(count)
    bar.finish()

    return seizure_data_dict


def load_edf_extract_seizures_v152(base_dir, save_data_dir, data_dict):
    seizure_data_dict = collections.defaultdict(list)


    count = 0
    bar = progressbar.ProgressBar(maxval=sum(len(v) for k, v in data_dict.items()),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for seizure_type, seizures in data_dict.items():
        for seizure in seizures:
            rel_file_location = seizure.filename.replace('.tse', '.edf').replace('./', 'edf/')
            patient_id = seizure.patient_id
            abs_file_location = os.path.join(base_dir, rel_file_location)
            try:
                temp = seizure_type_data(patient_id = patient_id, seizure_type = seizure_type, data = read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time))
                with open(os.path.join(save_data_dir, 'szr_' + str(count) + '_pid_' + patient_id + '_type_' + seizure_type + '.pkl'), 'wb') as fseiz:
                    pickle.dump(temp, fseiz)
                count += 1
            except Exception as e:
                print(e)
                print(rel_file_location)

            bar.update(count)
    bar.finish()

    return seizure_data_dict



def load_edf_extract_seizures_v200(base_dir, save_data_dir, data_dict):

    print("load edf_extract_seizures")
    print("base dir" , base_dir)    #base dir D:\v2.0.0\v2.0.0
    print("save_data_dir", save_data_dir) # D:\seizureergebnisse\v2.0.0\raw_seizures
    seizure_data_dict = collections.defaultdict(list)


    count = 0
    bar = progressbar.ProgressBar(maxval=sum(len(v) for k, v in data_dict.items()),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for seizure_type, seizures in data_dict.items():
        for seizure in seizures:
            rel_file_location = seizure.filename.replace('.csv_bi', '.edf').replace('./', 'edf/')
            patient_id = seizure.patient_id
            abs_file_location = os.path.join(base_dir, rel_file_location)
            try:
                temp = seizure_type_data(patient_id = patient_id, seizure_type = seizure_type, data = read_edfs_and_extract(abs_file_location, seizure.start_time, seizure.end_time))
                with open(os.path.join(save_data_dir, 'szr_' + str(count) + '_pid_' + patient_id + '_type_' + seizure_type + '.pkl'), 'wb') as fseiz:
                    pickle.dump(temp, fseiz)
                count += 1
            except Exception as e:
                print(e)
                print(rel_file_location)

            bar.update(count)


    bar.finish()

    return seizure_data_dict

def gen_raw_seizure_pkl_v200(args,tuh_eeg_szr_ver):
    print("gen_raw_seizure_pkl_v200")
                                 #tuh_eeg_szr_ver v2.0.0
    base_dir = args.base_dir     #(D:\v2.0.0)

    save_data_dir = os.path.join(args.save_data_dir, tuh_eeg_szr_ver, 'raw_seizures') #save directory D:\seizureergebnisse\v2.0.0\raw_seizures
    if not os.path.exists(save_data_dir):     #if dir doesnt exist create one
        os.makedirs(save_data_dir)

    raw_data_base_dir = os.path.join(base_dir, tuh_eeg_szr_ver, 'edf') # raw data_base_dir D:\v2.0.0\v2.0.0
    print("raw data base dir gen raw seizurev200", raw_data_base_dir)

    # For training files
    print('Parsing the seizures of the training set...\n')
    train_data_dict = generate_data_dict_v200(base_dir, 'train', tuh_eeg_szr_ver)
    print_type_information(train_data_dict)
    print('\n\n')

    print('Parsing the seizures of the validation set...\n')
    dev_test_data_dict = generate_data_dict_v200(base_dir, "dev", tuh_eeg_szr_ver)
    print('Number of seizures by type in the validation set...\n')
    print_type_information(dev_test_data_dict)
    print('\n\n')

    # Now we combine both
    print('Combining the training and validation set...\n')
    merged_dict = merge_train_test(dev_test_data_dict, train_data_dict)
    # merged_dict = merge_train_test(train_data_dict,dev_test_data_dict)
    print('Number of seizures by type in the combined set...\n')
    print_type_information(merged_dict)
    print('\n\n')

    seizure_data_dict = load_edf_extract_seizures_v200(raw_data_base_dir, save_data_dir, merged_dict)

    print_type_information(seizure_data_dict)
    print('\n\n')

# to convert raw edf data into pkl format raw data
def gen_raw_seizure_pkl(args,tuh_eeg_szr_ver, anno_file):
    base_dir = args.base_dir

    save_data_dir = os.path.join(args.save_data_dir, tuh_eeg_szr_ver, 'raw_seizures')
    if not os.path.exists(save_data_dir):
        os.makedirs(save_data_dir)

    raw_data_base_dir = os.path.join(base_dir, tuh_eeg_szr_ver)
    szr_annotation_file = os.path.join(raw_data_base_dir, '_DOCS', anno_file)

    # For training files
    print('Parsing the seizures of the training set...\n')
    train_data_dict = generate_data_dict(szr_annotation_file, 'train', tuh_eeg_szr_ver)
    print('Number of seizures by type in the training set...\n')
    print_type_information(train_data_dict)
    print('\n\n')

    # For dev files
    if tuh_eeg_szr_ver == 'v1.5.2':
        dev_name = 'dev'
    elif tuh_eeg_szr_ver == 'v1.4.0':
        dev_name = 'dev_test'
    elif tuh_eeg_szr_ver == 'v2.0.0':
        dev_name = 'dev'
    else:
        exit('tuh_eeg_szr_ver %s is not supported' % tuh_eeg_szr_ver)

    print('Parsing the seizures of the validation set...\n')
    dev_test_data_dict = generate_data_dict(szr_annotation_file, dev_name, tuh_eeg_szr_ver)
    print('Number of seizures by type in the validation set...\n')
    print_type_information(dev_test_data_dict)
    print('\n\n')

    # Now we combine both
    print('Combining the training and validation set...\n')
    merged_dict = merge_train_test(dev_test_data_dict, train_data_dict)
    # merged_dict = merge_train_test(train_data_dict,dev_test_data_dict)
    print('Number of seizures by type in the combined set...\n')
    print_type_information(merged_dict)
    for key in merged_dict:
        print (key)

    print('\n\n')

    # Extract the seizures from the edf files and save them
    if tuh_eeg_szr_ver == 'v1.5.2':
        seizure_data_dict = load_edf_extract_seizures_v152(raw_data_base_dir, save_data_dir, merged_dict)
    elif tuh_eeg_szr_ver == 'v1.4.0':
        seizure_data_dict = load_edf_extract_seizures_v140(raw_data_base_dir, save_data_dir, merged_dict)
    elif tuh_eeg_szr_ver == 'v2.0.0':
        seizure_data_dict = load_edf_extract_seizures_v140(raw_data_base_dir, save_data_dir, merged_dict)
    else:
        exit('tuh_eeg_szr_ver %s is not supported' % tuh_eeg_szr_ver)

    print_type_information(seizure_data_dict)
    print('\n\n')


def main():
    parser = argparse.ArgumentParser(description='Build data for TUH EEG data')

    if platform.system() == 'Linux':
        parser.add_argument('--base_dir', default='/slow1/raw_datasets/tuh/tuh_eeg_seizure/',
                            help='path to raw seizure dataset')
        parser.add_argument('--save_data_dir', default='/slow1/out_datasets/tuh/seizure_type_classification/',
                            help='path to save processed data')
    elif platform.system() == 'Darwin':
        parser.add_argument('--base_dir', default='/Users/jbtang/datasets/TUH/eeg_seizure/',
                            help='path to raw seizure dataset')
        parser.add_argument('--save_data_dir',
                            default='/Users/jbtang/datasets/TUH/output/seizures_type_classification/',
                            help='path to save processed data')
    elif platform.system() == 'Windows':
        parser.add_argument('--base_dir', default='/Users/jbtang/datasets/TUH/eeg_seizure/',
                            help='path to raw seizure dataset')
        parser.add_argument('--save_data_dir',
                            default='/Users/jbtang/datasets/TUH/output/seizures_type_classification/',
                            help='path to save processed data')

    else:
        print('Unknown OS platform %s' % platform.system())
        exit()

    parser.add_argument('-v', '--tuh_eeg_szr_ver',
                        default = 'v2.0.0',
                        #default='v1.5.2',
                        #default='v1.4.0',
                        help='version of TUH seizure dataset')

    args = parser.parse_args()
    tuh_eeg_szr_ver = args.tuh_eeg_szr_ver

    if tuh_eeg_szr_ver == 'v1.4.0': # for v1.4.0
        anno_file = 'seizures_v31r.xlsx'
        gen_raw_seizure_pkl(args, tuh_eeg_szr_ver, anno_file)
    elif tuh_eeg_szr_ver == 'v1.5.2': # for v1.5.2
        anno_file = 'seizures_v36r.xlsx'
        gen_raw_seizure_pkl(args, tuh_eeg_szr_ver, anno_file)
    elif tuh_eeg_szr_ver == 'v2.0.0':
        gen_raw_seizure_pkl_v200(args, tuh_eeg_szr_ver)
    else:
        exit('Not supported version number %s'%tuh_eeg_szr_ver)

if __name__ == '__main__':
    main()