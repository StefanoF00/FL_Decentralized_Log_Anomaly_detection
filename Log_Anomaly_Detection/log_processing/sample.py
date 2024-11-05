import pandas as pd
from collections import OrderedDict
import re
## Sampling and labeling function for BGL logs
def BGL_sampling(input_file, output_file, window_size, step_size, anomaly_threshold):
    '''
    Description:
      This function performs sliding window sampling on BGL logs to create
      labeled sequences based on specified parameters.
    Input parameters:
    - input_file: Path to the input CSV file containing BGL logs.
    - output_file: Path to save the resulting labeled sequences.
    - window_size: Size of the sliding window (number of consecutive logs considered).
    - step_size: Size of the step (increment) for the sliding window.
    - anomaly_threshold: Threshold for determining if a sequence is anomalous
                         based on the count of alert logs within the window.
    '''
    # Load CSV file
    data = pd.read_csv(input_file)

    # Count number of normal and abnormal logs
    # In the first column of the log, "-" indicates non-alert messages
    # while other strings (e.g. "APPREAD") are alert messages.
    total_logs = len(data)
    normal_logs = len(data[data['Label'] == '-'])
    anomaly_logs = total_logs - normal_logs
    print(f'Total number of logs: {total_logs}')
    print(f'Number of normal logs: {normal_logs}')
    print(f'Number of abnormal logs: {anomaly_logs}')

    print(f'Creating labeled sequences for {input_file}...')
    output_data = []

    # Sliding window
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[i:i+window_size]
        event_sequence = '[' + ', '.join(map(str, window['EventId'])) + ']'
        anomaly_count = len(window[window['Label'] != '-'])
        label = 'Anomaly' if anomaly_count >= anomaly_threshold else 'Normal'
        output_data.append([event_sequence, label])

    output_df = pd.DataFrame(output_data, columns=['EventSequence', 'Label'])
    output_df.to_csv(output_file, index=False)
    print('Done!')
    normal_count = len(output_df[output_df['Label'] == 'Normal'])
    anomaly_count = len(output_df[output_df['Label'] == 'Anomaly'])
    print(f'Total number of sequences: {normal_count+anomaly_count}')
    print(f'Number of normal sequences: {normal_count}')
    print(f'Number of abnormal sequences: {anomaly_count}\n')

## Sampling and labeling function for  HDFS logs
def HDFS_sampling(imput_file, label_file, output_file):
    '''
    Description:
      This function performs sampling and labeling of HDFS logs to create
      labeled sequences based on provided labels.
    Input parameters:
    - imput_file: Path to the HDFS log file.
    - label_file: Path to the CSV file containing labels for anomaly detection.
    - output_file: Path to save the resulting labeled sequences.
    '''

    print(f'* Creating labeled sequences for {imput_file}...', end='')
    struct_log = pd.read_csv(imput_file, engine='c', na_filter=False, memory_map=True)
    # mapping
    data_dict = OrderedDict()
    for idx, row in struct_log.iterrows():
        blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if not blk_Id in data_dict:
                data_dict[blk_Id] = []
            data_dict[blk_Id].append(row['EventId'])

    data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
    labels = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
    labeled_sequence = pd.merge(data_df, labels, on='BlockId')[['EventSequence', 'Label']]
    labeled_sequence.to_csv(output_file, index=False)
    print('Done!')
    normal_count = len(labeled_sequence[labeled_sequence['Label'] == 'Normal'])
    anomaly_count = len(labeled_sequence[labeled_sequence['Label'] == 'Anomaly'])
    total_count = normal_count + anomaly_count
    print(f'  - Total number of sequences: {total_count}')
    print(f'  - Number of normal sequences: {normal_count}')
    print(f'  - Number of abnormal sequences: {anomaly_count}\n')

## Function for data splitting into train and test sets
def split_train_test_logs(i, input_csv, training_csv, test_csv, validation_csv, n_train_normal, n_train_abnormal, n_test_normal, n_test_abnormal, clients):
    '''
    Description:
      Splits the input CSV into training and test datasets and save them to separate CSV files
    Input parameters:
    - input_csv: Path to the input CSV file.
    - training_csv: Path to save the training dataset CSV file.
    - test_csv: Path to save the test dataset CSV file for the current client.
    - n_train: Number of samples to include in the training dataset for each class.
    - n_test_normal: Number of normal samples to include in the test dataset for each client.
    - n_test_abnormal: Number of abnormal samples to include in the test dataset for each client.
    '''
    print(f'* Creating balanced train and test dataset from {input_csv}...')
    # Load the CSV file
    data = pd.read_csv(input_csv)

    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split the data into normal and anomaly sequences
    normal_data = data[data['Label'] == 'Normal']
    anomaly_data = data[data['Label'] == 'Anomaly']

    # Take the first n samples for training data
    training_data = pd.concat([normal_data[:n_train_normal], anomaly_data[:n_train_abnormal]])
    print(f'  - Samples in train dataset: {len(training_data)}')

    # Take the the samples for testing data
    test_data = pd.concat(
        [
            normal_data[n_train_normal : n_train_normal + n_test_normal],\
            anomaly_data[n_train_abnormal : n_train_abnormal + n_test_abnormal]
        ]
    )
    print(f'  - Samples in test dataset: {len(test_data)}')

    if i == 1:
        # initialize the validation dataset if this is the 1st iteration of split_train_test_logs()
        validation_data = pd.DataFrame()
    else:
        validation_data = pd.read_csv(validation_csv)
        
    remaining_data = pd.concat(
        [
            normal_data[n_train_normal + n_test_normal:],
            anomaly_data[n_train_abnormal + n_test_abnormal:]
        ]
    )    
    combined_data = pd.concat([validation_data, remaining_data])
    
    print(f'  - Total unused samples: {len(combined_data)}:')
    print(f'    - Added {len(normal_data[n_train_normal + n_test_normal:])} normal samples remaining')
    print(f'    - Added {len(anomaly_data[n_train_abnormal + n_test_abnormal:])} abnormal samples remaining\n')
    
    if i == clients:
        # Define a smaller validation data if this is the last iteration of split_train_test_logs()
        print('* Creating a validation dataset from the unused data...')
        normal_data = combined_data[combined_data['Label'] == 'Normal'].sample(frac=1).reset_index(drop=True)
        anomaly_data = combined_data[combined_data['Label'] == 'Anomaly']
        print(f'  - There are {len(normal_data)} unused normal samples')
        print(f'  - There are {len(anomaly_data)} unused abnormal samples')
        final_validation_data = pd.concat([normal_data[:len(anomaly_data)*5], anomaly_data])
        print(f'  - Validation dataset is composed by {len(final_validation_data)} samples')
    # Write the training and test data to CSV files
    training_data.to_csv(training_csv, index=False)
    test_data.to_csv(test_csv, index=False)
    if i == 5:
        final_validation_data.to_csv(validation_csv, index=False)
    else:
        combined_data.to_csv(validation_csv, index=False)