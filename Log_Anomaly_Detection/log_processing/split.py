import pandas as pd
import re
import os
from collections import OrderedDict

# HDFS functions
def split_HDFS_raw_logs(logs, label_file, normal_logs_file, anomaly_logs_file):
    '''
    Description:
      This function splits raw HDFS logs into normal and abnormal logs
      based on provided anomaly labels.
    Inputs parameters:
    - logs: Path to the raw HDFS logs file.
    - label_file: Path to the CSV file containing anomaly labels for the logs.
    - normal_logs_file: Path to save the normal logs after splitting.
    - anomaly_logs_file: Path to save the abnormal (anomaly) logs after splitting.
    '''
    anomaly_data = pd.read_csv(label_file)
    anomaly_block_ids = set(anomaly_data[anomaly_data['Label'] == 'Anomaly']['BlockId'].astype(str))
    print(f'there are {len(anomaly_block_ids)} abnormal block indices')

    with open(normal_logs_file, 'w') as normal_file, open(anomaly_logs_file, 'w') as anomaly_file:
        with open(logs, 'r') as logs_file:
            for log_entry in logs_file:
                block_id = extract_block_id(log_entry)
                if block_id and block_id in anomaly_block_ids:
                    anomaly_file.write(log_entry)
                else:
                    normal_file.write(log_entry)

    print(f'Number of normal logs: {count_lines(normal_logs_file)}')
    print(f'Number of abnormal logs: {count_lines(anomaly_logs_file)}')

def extract_block_id(log_entry):
    '''
    Description:
      This function extracts the block id from a raw HDFS log entry,
      useful to determine the session and the label of the analyzed entry
    Inputs parameters:
    - log_entry: HDFS log line
    '''
    match = re.search(r'(blk_-?\d+)', log_entry)
    if match:
        return match.group(0)
    else:
        return None

# BGL function:
def split_BGL_raw_logs(logs, normal_logs_file, anomaly_logs_file):
    '''
    Description:
      This function splits raw BGL logs into normal and abnormal logs
      based on provided anomaly labels.
    Inputs parameters:
    - logs: Path to the raw HDFS logs file.
    - normal_logs_file: Path to save the normal logs after splitting.
    - anomaly_logs_file: Path to save the abnormal (anomaly) logs after splitting.
    '''
    with open(normal_logs_file, 'w') as normal_file, open(anomaly_logs_file, 'w') as anomaly_file:
        with open(logs, 'r') as logs_file:
            for log_entry in logs_file:
                label = log_entry.strip().split()[0]
                if label != '-':
                    anomaly_file.write(log_entry)
                else:
                    normal_file.write(log_entry)

    print(f'Number of normal logs: {count_lines(normal_logs_file)}')
    print(f'Number of abnormal logs: {count_lines(anomaly_logs_file)}')

# universal functions:
def count_lines(filename):
    with open(filename, 'r') as file:
        return sum(1 for line in file)

def create_clients_data(normal_logs_file, anomaly_logs_file, output_directory, k):
    '''
    Description:
      This function creates client-specific log files from split normal and anomaly logs
      to simulate data distribution for federated learning.
    Inputs parameters:
    - normal_logs_file: Path to the file containing normal logs.
    - anomaly_logs_file: Path to the file containing anomaly logs.
    - output_directory: Directory to save the client-specific log files.
    - k: Number of client log files to create.
    '''
    total_normal_lines = count_lines(normal_logs_file)
    total_anomaly_lines = count_lines(anomaly_logs_file)
    normal_lines_per_file = total_normal_lines // k
    anomaly_lines_per_file = total_anomaly_lines // k

    with open(normal_logs_file, 'r') as normal_file:
        normal_lines = normal_file.readlines()

    with open(anomaly_logs_file, 'r') as anomaly_file:
        anomaly_lines = anomaly_file.readlines()

    for i in range(k):
        print(f'creating client {i+1}...')
        output_file_path = f'{output_directory}client_{i+1}.log'
        with open(output_file_path, 'w') as output_file:
            start_index_normal = i * normal_lines_per_file
            end_index_normal = start_index_normal + normal_lines_per_file
            for line in normal_lines[start_index_normal:end_index_normal]:
                output_file.write(line)

            start_index_anomaly = i * anomaly_lines_per_file
            end_index_anomaly = start_index_anomaly + anomaly_lines_per_file
            for line in anomaly_lines[start_index_anomaly:end_index_anomaly]:
                output_file.write(line)
            print(f'created {output_file_path}: it has {count_lines(output_file_path)} logs')
