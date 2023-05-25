def gen_raw_seizure_pkl_v200(args,tuh_eeg_szr_ver):
    print("gen_raw_seizure_pkl_v200")
                                 #tuh_eeg_szr_ver v2.0.0
    base_dir = args.base_dir     #(D:\v2.0.0)

    save_data_dir = os.path.join(args.save_data_dir, tuh_eeg_szr_ver, 'raw_seizures') #save directory D:\seizureergebnisse\v2.0.0\raw_seizures
    if not os.path.exists(save_data_dir):     #if dir doesnt exist create one
        os.makedirs(save_data_dir)

    raw_data_base_dir = os.path.join(base_dir, tuh_eeg_szr_ver) # raw data_base_dir D:\v2.0.0\v2.0.0

    # For training files
    print('Parsing the seizures of the training set...\n')









    # For dev files
    if tuh_eeg_szr_ver == 'v2.0.0':
        dev_name = 'dev'
    else:
        exit('tuh_eeg_szr_ver %s is not supported' % tuh_eeg_szr_ver)


        
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