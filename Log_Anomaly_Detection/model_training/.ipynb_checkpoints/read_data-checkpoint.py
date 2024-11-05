import pandas as pd
import numpy as np
import json
import torch
import time

def split_train_data(data_path, split):
    '''
    Description:
      function that splits a dataset into train & validation sets
    Input parameters:
    - data_path: csv file of the train dataset to be splitted
    - split: parameter specifing the proportion of the dataset
             allocated for training
    Outputs:
    - training_data: Pandas DataFrame containing training dataset
    - valid_data: Pandas DataFrame containing validation dataset
    '''
    # Read the CSV file
    data = pd.read_csv(data_path) # 27000
    
    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split the data into normal and anomaly sequences
    normal_data = data[data['Label'] == 'Normal'] #24000
    anomaly_data = data[data['Label'] == 'Anomaly'] #3000

    n_train = int(len(normal_data)*split) # 21600
    n_valid = int(len(normal_data)*(1-split)) #2399

    n_train_normal = int(n_train *0.9)
    n_train_abnormal = int(n_train * 0.1)
    
    # Take the first n_train samples for training data
    training_data = pd.concat([normal_data[:n_train_normal], anomaly_data[:n_train_abnormal]])

    valid_data = pd.concat(
        [
            normal_data[n_train_normal : n_train_normal + n_valid],\
            anomaly_data[n_train_abnormal : n_train_abnormal + n_valid]
        ]
    )
    return training_data, valid_data

def get_session(EventSequence):
    '''
    Description:
      function that extracts int log sequences in a string representing a list
    Input parameter:
    - EventSequence: raw string whose characters are numbers, spaces and square bracket
                     e.g: '[ 21, 35, 35, 11, 4, 7 ]'
    Output:
    - session: list of integers, representing log keys
    '''
    session = EventSequence.strip('[]').split(',')
    session = [log.strip() for log in session]
    session = [int(log) for log in session]
    return session

def get_ppa_logs(logs, templates_PPA):
    '''
    Description:
      function that maps each log entry to its corresponding
      semantic vector within a set of log sessions.
    Input parameters:
    - logs: set of log sessions (i.e. matrix of integers, each one representing a log).
    - templates_PPA: csv file with column 'EmbeddingPPA', containing JSON vectors that
      represent the embeddings of the corresponding log key in the column 'EventId'.
    Output:
    - PPA_logs: matrix such that each element represents the log key
                as its correspondent semantic vector
    '''
    df = pd.read_csv(templates_PPA)

    # ppa_result: List in which will be stored all the embeddings, organized such that:
    # (j-1)-th embedding corresponds to j-th log key ∀j=1,...,num_log_keys_for_client_i
    ppa_result = []
    for embedding in df["EmbeddingPPA"]:
        vec = np.array(json.loads(embedding))
        ppa_result.append(vec)
    ppa_result = np.array(ppa_result)
    dim = ppa_result.shape[1]
    PPA_logs = []
    for i in range(0,len(logs)):
        padding = np.zeros((300,dim))
        session = get_session(logs[i])
        for j in range(0,len(session)):
            padding[j] = ppa_result[session[j]-1]
        padding = list(padding)
        PPA_logs.append(padding)
    PPA_logs = np.array(PPA_logs)
    return PPA_logs

def read_train_data(data_path, templates_PPA, split=0.9):
    '''
    Description:
      function that reads the samples and labels for model training,
      dividing them into 2 subset, one for training, one for validation
    Input parameters:
    - data_path: training dataset, csv file with 2 columns (EventTemplate, Label)
    - templates_PPA: csv file with column 'EmbeddingPPA', containing JSON vectors that
      represent the embeddings of the corresponding log key in the column 'EventId'.
    - device: cuda, cpu, mps
    Outputs:
    - x_train: training samples
    - y_train: training labels
    - x_valid: validation samples
    - y_valid: validation labels
    '''
    # Split the logs in train & validation sets
    train_data, valid_data = split_train_data(data_path, split)

    # Get log and label columns from train dataset
    train_data = train_data.values
    train_logs = train_data[:, 0]
    train_labels = train_data[:, 1]

    # Get log and label columns from valid dataset
    valid_data = valid_data.values
    valid_logs = valid_data[:, 0]
    valid_labels = valid_data[:, 1]

    # Compute from each log key column the equivalent PPA semantic vectorm column
    # ie find the train and valid data
    x_train = get_ppa_logs(train_logs, templates_PPA)
    x_valid = get_ppa_logs(valid_logs, templates_PPA)
    x_train = torch.tensor(x_train, dtype=torch.float32)  # Assuming x_train is of type float32
    x_valid = torch.tensor(x_valid, dtype=torch.float32)

    # Compute the train and valid labels in one-hot-encoding
    y_train = torch.tensor([0 if label == 'Normal' else 1 for label in train_labels])
    y_valid = torch.tensor([0 if label == 'Normal' else 1 for label in valid_labels])
    #y_train = torch.nn.functional.one_hot(y_train, num_classes=2)
    #y_valid = torch.nn.functional.one_hot(y_valid, num_classes=2)

    return x_train, y_train, x_valid, y_valid

def read_test_data(path, templates_PPA):
    '''
    Description:
      function that reads the samples and labels for model testing
    Inputs:
    - path: test dataset, csv file with 2 columns (EventTemplate, Label)
    - templates_PPA: csv file with column 'EmbeddingPPA', containing JSON vectors that
      represent the embeddings of the corresponding log key in the column 'EventId'.
    Outputs:
    - x_test: testing samples
    - y_test: testing labels
    '''
    # Get log and label columns from test dataset
    test_data = pd.read_csv(path)
    test_data = test_data.values
    test_logs = test_data[:,0]
    test_labels = test_data[:,1]

    # Compute from each log key column the equivalent PPA semantic vectorm column
    # ie find the test data
    x_test = get_ppa_logs(test_logs, templates_PPA)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor([0 if label == 'Normal' else 1 for label in test_labels])
    #y_test = torch.nn.functional.one_hot(y_test, num_classes=2)
    return x_test, y_test # they must be on CPU device

def read(K, processed_data_dir, embedded_data_dir, processed_validation_data, device, split):
    # Initialize data structures as lists
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    x_valid_glob = []
    y_valid_glob = []
    x_test = []
    y_test = []

    start_total = time.perf_counter()
    for i in range(0, K):
        print(f'\n* Client {i+1}')
        print(f'  - Reading train and validation data of client {i+1}...', end='')

        # Define the paths
        processed_client_data_dir = processed_data_dir + f'client_data_{i+1}/'
        processed_client_data_train = processed_client_data_dir + f'client_{i+1}_train.csv'
        processed_client_data_test = processed_client_data_dir + f'client_{i+1}_test.csv'
        embedded_client_data_dir = embedded_data_dir + f'client_data_{i+1}/'
        embedded_PPA_client_data = embedded_client_data_dir + f'client_{i+1}.log_embedding_ppa.csv'
        
        start_data = time.perf_counter()
        x_train_client, y_train_client, x_valid_client, y_valid_client = read_train_data(
            processed_client_data_train,
            embedded_PPA_client_data,
            split
        )
        end_data = time.perf_counter()
        print(f'Done! Performed in {end_data - start_data:.4f}s!')

        # Append to lists
        x_train.append(x_train_client)
        y_train.append(y_train_client)
        x_valid.append(x_valid_client)
        y_valid.append(y_valid_client)
        
        # Read and process test data for the current client
        print(f'  - Reading test data of client {i+1}...', end='')
        start_data = time.perf_counter()
        x_test_client, y_test_client = read_test_data(
            processed_client_data_test,
            embedded_PPA_client_data
        )
        end_data = time.perf_counter()
        print(f'Done! Performed in {end_data - start_data:.4f}s!')

        # Append to lists
        x_test.append(x_test_client)
        y_test.append(y_test_client)
    print('\n* Reading model validation data...', end='')
    x_valid_glob, y_valid_glob = read_test_data(
        processed_validation_data,
        embedded_data_dir + 'client_data_4/client_4.log_embedding_ppa.csv',
    )
    print('Done!')

    # Convert lists of tensors to single tensors
    x_train = torch.stack(x_train).to(device)
    y_train = torch.stack(y_train).to(device)

    x_valid = torch.stack(x_valid).to(device)
    y_valid = torch.stack(y_valid).to(device)

    #x_test = torch.cat(x_test, dim=0)
    #y_test = torch.cat(y_test, dim=0)

    end_total = time.perf_counter()
    print(f'\nAll the data are successfully read and stored in variables in {end_total - start_total:.4f}s!')
    samples_per_client = [x_train[i].shape[0] for i in range(len(x_train))]    
    print(f'samples per client: {samples_per_client}')

    return x_train, y_train, x_valid, y_valid, x_valid_glob, y_valid_glob, x_test, y_test