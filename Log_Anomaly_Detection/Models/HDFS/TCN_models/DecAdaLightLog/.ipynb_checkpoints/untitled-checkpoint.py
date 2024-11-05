import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import os
from consensus.simulate_integrator import simulate_network, compute_T, plot_integrator_M_controls


# TCN MODEL DEFINITION
class ResBlock(nn.Module):
    '''
    Description:
      The ResBlock class implements a residual block used in CNNs to improve
      network depth while mitigating the vanishing gradient problem.
      This block is designed for use within a Time Convolutional Network (TCN).
      The ResBlock comprises two 1D convolutional layers with ReLU
      activation and a shortcut connection.
      - Layer 1: convolves the input with specified parameters, followed by ReLU.
      - Layer 2: convolves the output of the first layer with the same number
        of output channels & kernel size, followed by ReLU activation.
      - Shortcut: adapts based on the input and output tensor sizes:
        if they match, an identity shortcut connection (nn.Identity()) is used;
        otherwise, an additional 1D convolution (nn.Conv1d) adjusts the sizes.
        The final output results from summing the shortcut connection output
        and the last convolutional layer output, followed by a ReLU activation.

    Inputs:
    - in_channels: Number of input channels (size of the input).
    - out_channels: Number of output channels (number of filters).
    - kernel_size: Size of the convolutional kernel.
    - dilation_rate: Dilation rate for the convolution.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', dilation=dilation_rate)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)  # Modifica qui in_channels a out_channels
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same')

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(residual)
        residual = self.conv2(residual)
        residual = F.relu(residual)
        shortcut = self.shortcut(x)
        output = shortcut + residual
        output = F.relu(output)
        return output

class TCN_model(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TCN_model, self).__init__()
        self.resblock1 = ResBlock(input_size, out_channels=3, kernel_size=3, dilation_rate=1)
        self.resblock2 = ResBlock(3, out_channels=3, kernel_size=3, dilation_rate=2)
        self.resblock3 = ResBlock(3, out_channels=3, kernel_size=3, dilation_rate=4)
        self.resblock4 = ResBlock(3, out_channels=3, kernel_size=3, dilation_rate=8)
        self.global_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(3, num_classes)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.global_pooling(x)
        x = x.squeeze(2)
        x = self.fc(x)
        return x

# TRAIN FUNCTION DEFINITIONS

def calculate_loss(model, data_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            #loss = criterion(outputs, labels.float())
            # outputs are the predictions of the data labels
            # labels are the true labels
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    return running_loss / len(data_loader)

def train_local_model(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            #loss = criterion(outputs, labels.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
    return model

def evaluate_train_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # Ottieni le predizioni
            _, predicted = torch.max(outputs, 1)
            # Converte le etichette one-hot in indici di classe
            #_, labels = torch.max(labels, 1)
            # Calcola il numero di predizioni corrette
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calcola l'accuratezza
    accuracy = 100 * correct / total
    return accuracy
    
def get_validation_metrics(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(valid_loader)
    epoch_accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_predicted,zero_division=0)
    recall = recall_score(all_labels, all_predicted,zero_division=0)
    f1 = f1_score(all_labels, all_predicted,zero_division=0)
    all_labels = torch.tensor(all_labels)
    all_predicted = torch.tensor(all_predicted)
    TP = torch.sum((all_labels == 1) & (all_predicted == 1))
    FP = torch.sum((all_labels == 0) & (all_predicted == 1))
    TN = torch.sum((all_labels == 0) & (all_predicted == 0))
    FN = torch.sum((all_labels == 1) & (all_predicted == 0))
    #print(f'  - TN: {TN} | FP: {FP} | TP: {TP} | FN: {FN}')
    return epoch_loss, epoch_accuracy, precision*100, recall*100, f1*100, TP, FP, TN, FN

def evaluate_clients_scores(score, models, num_clients, criteria, x, y, device):
    scores = []
    valid_dataset = TensorDataset(x, y)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
    scores = []
    print(' - Evaluating models o a common validation dataset:')
    for i in range(num_clients): 
        val_loss, val_accuracy, val_precision, val_recall, val_f1, TP, FP, TN, FN = get_validation_metrics(
            models[i],\
            valid_loader,\
            criteria[i],\
            device
        )
        print(f'   ⮑ Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%, Prec: {val_precision:.2f}%, Rec: {val_recall:.2f}%, F1: {val_f1:.2f}%')
        if score == 'f1':
            metric = val_f1
        elif score == 'precision':
            metric = val_precision
        elif score == 'accuracy':
            metric = val_accuracy
        else:
            metric = val_recall
        scores.append(metric/100)
    formatted_scores = [f"{s:.4f}" for s in scores]
    print(f'   ⮑ Scores: {formatted_scores}')
    return scores
    
def average_model_weights(models):
    '''
    for FedAvg
    '''
    sd = [models[i].state_dict() for i in range(len(models))]
    sd_global = {} #MODIFICATO QUI DA EDS
    for key in sd[0]:
        total = 0
        for k in range(0,len(sd)):
            total+= sd[k][key]
        sd_global[key] = total / len(sd)
    for model in models:
        model.load_state_dict(sd_global)
    return sd_global

def average_weighted_model_weights(models, scores):
    '''
    For AdaFed: weighted average based on obtained scores for each client
    Weights are normalized (their sum is 1)
    '''
    weights = [score / sum(scores) for score in scores]
    formatted_weights = [f"{weight:.4f}" for weight in weights]
    print(f"   ⮑ Weights: {formatted_weights}")
    sd = [models[i].state_dict() for i in range(len(models))]

    sd_global = {}
    for key in sd[0]:
        total = 0
        for k in range(len(sd)):
            total += sd[k][key] * weights[k]
        # No need to divide sd_global[key] by sum of weights since they are normalized
        sd_global[key] = total
    for model in models:
        model.load_state_dict(sd_global)
    return sd_global

def decentralized_average_model_weights(models, consensus_params):
    '''
    for DecFedAvg
    '''
    # Get all the parameters for consensus
    k1 = consensus_params.get("k1")     # Initial gain for v_i
    k2 = consensus_params.get("k2")     # Initial gain for tanh term
    delta = consensus_params.get("delta")
    alpha = consensus_params.get("alpha")  # Exponent for v_i (radio of pos ODD numbers)
    c = consensus_params.get("c")
    theta = consensus_params.get("theta")
    delta1 = consensus_params.get("delta1") # Saturation threshold
    dt = consensus_params.get("dt")    # Time step size
    dist = consensus_params.get("dist")
    integrator = consensus_params.get("integrator")
    integrator_lemma2 = consensus_params.get("integrator_lemma2")
    A = consensus_params.get("A")
    N = len(models)-1
    
    # List containing the state_dict of the N models
    sd = [models[i].state_dict() for i in range(N)]
    M = sum(param.numel() for param in sd[0].values())  # Number of parameter groups
    # Initialize a list to store flattened parameters for each model
    flattened_params = []
    
    # Flatten the parameters of each model and store them
    for i in range(N):
        model_sd = sd[i]
        flattened = np.concatenate([param.detach().cpu().numpy().flatten() for param in model_sd.values()])
        flattened_params.append(flattened)
    initial_states = np.array(flattened_params)
    '''
    T_m = compute_T(integrator_lemma2, initial_states, alpha, c, 0.9)

    # Vector of M time intervals (m-th elem is the biggest one for the m-th consensus protocol)
    T = np.zeros(M)
    for m in range(M):
        T[m] = int(np.max(T_m[m])+1)
    '''
    T=8
    # Simulate M consensus protocols
    final_state, trajectory, time_values = simulate_network(N, M, T, dt, A, k1, k2, alpha, delta1, initial_states, integrator)

    # Compute the consensuses
    consensus = trajectory[-1, :, :] # ora è diverso per ogni peso e per ogni nodo (prima era diverso solo per ogni peso)
    print(trajectory.shape)
    # Load consensus values as final weights 
    sd_global = {}
    start_idx = 0
    for key in sd[0].keys():
        param_shape = sd[0][key].shape
        param_size = np.prod(param_shape)
        
        # Extract the consensus weights for this parameter
        consensus_weights = consensus[start_idx:start_idx + param_size]
        
        # Reshape the weights to their original shape
        sd_global[key] = torch.tensor(consensus_weights).reshape(param_shape)
        start_idx += param_size

    return sd_global

def decentralized_average_model_weights(models, consensus_params):
    # Get all the parameters for consensus
    k1 = consensus_params.get("k1")     # Initial gain for v_i
    k2 = consensus_params.get("k2")     # Initial gain for tanh term
    delta = consensus_params.get("delta")
    alpha = consensus_params.get("alpha")  # Exponent for v_i (radio of pos ODD numbers)
    c = consensus_params.get("c")
    theta = consensus_params.get("theta")
    delta1 = consensus_params.get("delta1") # Saturation threshold
    dt = consensus_params.get("dt")    # Time step size
    dist = consensus_params.get("dist")
    integrator = consensus_params.get("integrator")
    integrator_lemma2 = consensus_params.get("integrator_lemma2")
    A = consensus_params.get("A")
    N = len(models)
    
    # List containing the state_dict of the N models
    sd = [models[i].state_dict() for i in range(N)]
    M = sum(param.numel() for param in sd[0].values())  # Number of parameter groups
    # Initialize a list to store flattened parameters for each model
    flattened_params = []
    
    # Flatten the parameters of each model and store them
    for i in range(N):
        model_sd = sd[i]
        flattened = np.concatenate([param.detach().cpu().numpy().flatten() for param in model_sd.values()])
        flattened_params.append(flattened)
    initial_states = np.array(flattened_params)

    '''
    # IN QUESTO MODO UN'EPOCA GLOBALE DURA CIRCA 4 MINUTI 
    T_m = compute_T(integrator_lemma2, initial_states, alpha, c, 0.9)

    # Vector of M time intervals (m-th elem is the biggest one for the m-th consensus protocol)
    T = np.zeros(M)
    for m in range(M):
        T[m] = int(np.max(T_m[m])+1)
    print(T) 
    '''
    T=200
    # Simulate M consensus protocols
    print(f" - Starting decentralized model averaging: executing M={M} consensus protocols (one for each parameter group).")
    print("   ⮑ Each client has same Weights (influence on the model averaging)")
    final_state, trajectory, time_values = simulate_network(N, M, T, dt, A, k1, k2, alpha, delta1, initial_states, integrator)
    '''
    # Compute the weighted consensus as the weighted mean of the last state across all nodes
    weighted_consensus = np.average(trajectory[-1, :, :], axis=0, weights=weights)

    # Load consensus values as final weights 
    sd_global = {}
    start_idx = 0
    for key in sd[0].keys():
        param_shape = sd[0][key].shape
        param_size = np.prod(param_shape)
        
        # Extract the consensus weights for this parameter
        consensus_weights = weighted_consensus[start_idx:start_idx + param_size]
        
        # Reshape the weights to their original shape
        sd_global[key] = torch.tensor(consensus_weights).reshape(param_shape)
        
        start_idx += param_size

    return sd_global
    '''
    #trajectory_node_1 = []
    # List to store the updated state_dict for each node
    updated_sd_list = []
    #trajectory has shape (T,N,M)
    # Iterate over each node to update its weights using the final state of its trajectory
    for n in range(N):
        node_final_state = trajectory[-1, n, :]  # Final M states for node n (shape = (,M))
        print(node_final_state[:3])
        #trajectory_node_1.append(trajectory[:,n,0])
        
        start_idx = 0
        sd_local = {}
        #print(node_final_state[:3]) 
        for key in sd[0].keys():  # Assuming all state_dicts have the same structure
            param_shape = sd[0][key].shape
            param_size = np.prod(param_shape)
            
            # Extract the weighted consensus weights for this parameter
            consensus_weights = node_final_state[start_idx:start_idx + param_size]
            
            # Reshape the weights to their original shape and store in sd_local
            sd_local[key] = torch.tensor(consensus_weights).reshape(param_shape)
            
            start_idx += param_size
        
        # Add the updated state dict for this node to the list
        updated_sd_list.append(sd_local)
    #print(np.array(trajectory_node_1).shape)
    #plot_integrator_M_controls(np.array(trajectory_node_1),time_values)
    # Now, each node has an updated set of weights in updated_sd_list
    
    # Optionally, update the models with these new weights
    for i in range(N):
        models[i].load_state_dict(updated_sd_list[i])
    
    # Return the updated state_dict for one of the nodes, or for further use in a decentralized way
    return updated_sd_list
    
def decentralized_average_model_weights(models, consensus_params):
    #weights = np.array([score / sum(scores) for score in scores])
    #formatted_weights = [f"{weight:.4f}" for weight in weights]
    
    # Get all the parameters for consensus
    k1 = consensus_params.get("k1")     # Initial gain for v_i
    k2 = consensus_params.get("k2")     # Initial gain for tanh term
    delta = consensus_params.get("delta")
    alpha = consensus_params.get("alpha")  # Exponent for v_i (radio of pos ODD numbers)
    c = consensus_params.get("c")
    theta = consensus_params.get("theta")
    delta1 = consensus_params.get("delta1") # Saturation threshold
    dt = consensus_params.get("dt")    # Time step size
    dist = consensus_params.get("dist")
    integrator = consensus_params.get("integrator")
    integrator_lemma2 = consensus_params.get("integrator_lemma2")
    #A = np.array(consensus_params.get("A"),dtype=np.float64)
    A = consensus_params.get("A")
    N = len(models)
    
    # List containing the state_dict of the N models
    sd = [models[i].state_dict() for i in range(N)]
    M = sum(param.numel() for param in sd[0].values())  # Number of parameter groups
    # Initialize a list to store flattened parameters for each model
    flattened_params = []
    
    # Flatten the parameters of each model, weight and store them
    for i in range(N):
        model_sd = sd[i]
        flattened = np.concatenate([param.detach().cpu().numpy().flatten() for param in model_sd.values()])
        flattened_params.append(flattened)
    initial_states = np.array(flattened_params)

    '''
    # IN QUESTO MODO UN'EPOCA GLOBALE DURA CIRCA 4 MINUTI 
    T_m = compute_T(integrator_lemma2, initial_states, alpha, c, 0.9)

    # Vector of M time intervals (m-th elem is the biggest one for the m-th consensus protocol)
    T = np.zeros(M)
    for m in range(M):
        T[m] = int(np.max(T_m[m])+1)
    print(T) 
    '''
    T=200 # prova con dt = 0.1 e T = 40
    # Weighted Average: multiply each row by the corresponding node's weight
    # In case of Simple average, the weights are all the same
    #A *= weights[:, None]
    #print(f' - Adj matrix:{A}')
    # Simulate M consensus protocols
    '''
    if weighted_avg:
        print(f" - Starting decentralized weighted model averaging: executing M={M} consensus protocols (one for each parameter group).")
        print(f"   ⮑ Weights for each client: {formatted_weights}")
    else:
        print(f" - Starting decentralized model averaging: executing M={M} consensus protocols (one for each parameter group).")
    '''
    print(f" - Starting decentralized model averaging: executing M={M} consensus protocols (one for each parameter group).")
    final_state, trajectory, time_values = simulate_network(N, M, T, dt, A, k1, k2, alpha, delta1, initial_states, integrator)
    '''
    # Compute the weighted consensus as the weighted mean of the last state across all nodes
    weighted_consensus = np.average(trajectory[-1, :, :], axis=0, weights=weights)

    # Load consensus values as final weights 
    sd_global = {}
    start_idx = 0
    for key in sd[0].keys():
        param_shape = sd[0][key].shape
        param_size = np.prod(param_shape)
        
        # Extract the consensus weights for this parameter
        consensus_weights = weighted_consensus[start_idx:start_idx + param_size]
        
        # Reshape the weights to their original shape
        sd_global[key] = torch.tensor(consensus_weights).reshape(param_shape)
        
        start_idx += param_size

    return sd_global
    '''
    #trajectory_node_1 = []
    # List to store the updated state_dict for each node
    updated_sd_list = []
    #trajectory has shape (T,N,M)
    # Iterate over each node to update its weights using the final state of its trajectory
    for n in range(N):
        node_final_state = trajectory[-1, n, :]  # Final M states for node n (shape = (,M))
        print(node_final_state[:3])
        #trajectory_node_1.append(trajectory[:,n,0])
        
        start_idx = 0
        sd_local = {}
        #print(node_final_state[:3]) 
        for key in sd[0].keys():  # Assuming all state_dicts have the same structure
            param_shape = sd[0][key].shape
            param_size = np.prod(param_shape)
            
            # Extract the weighted consensus weights for this parameter
            consensus_weights = node_final_state[start_idx:start_idx + param_size]
            
            # Reshape the weights to their original shape and store in sd_local
            sd_local[key] = torch.tensor(consensus_weights).reshape(param_shape)
            
            start_idx += param_size
        
        # Add the updated state dict for this node to the list
        updated_sd_list.append(sd_local)
    #print(np.array(trajectory_node_1).shape)
    #plot_integrator_M_controls(np.array(trajectory_node_1),time_values)
    # Now, each node has an updated set of weights in updated_sd_list
    
    # Optionally, update the models with these new weights
    for i in range(N):
        models[i].load_state_dict(updated_sd_list[i])
    
    # Return the updated state_dict for one of the nodes, or for further use in a decentralized way
    return updated_sd_list

    
def update_loss_weights_f1_score(f1_score, epsilon=0.1):
    '''
    Description
      Updates weights using f1 score such that they can assume values
      in range [0,10]
    '''
    K_abnormal = 1 / (f1_score + epsilon)
    K_normal = 1 / ((1 - f1_score) + epsilon)
    return K_normal, K_abnormal
    
def update_loss_weights_recall(recall, epsilon=0.1):
    '''
    Description
      Updates weights using recall such that they can assume values
      in range [0,10]
    '''
    K_abnormal = 1 / (recall + epsilon)
    K_normal = 1 / ((1 - recall) + epsilon)
    return K_normal, K_abnormal
    
def update_loss_weights_precision(precision, epsilon=0.1):
    '''
    Description
      Updates weights using recall such that they can assume values
      in range [0,10]
    '''
    K_abnormal = 1 / (precision + epsilon)
    K_normal = 1 / ((1 - precision) + epsilon)
    return K_normal, K_abnormal
    
def update_loss_weights(metric, epsilon=0.1):
    '''
    Description
      Updates weights using recall such that they can assume values
      in range [0,10]
    '''
    K_abnormal = 1 / (metric + epsilon)
    K_normal = 1 / ((1 - metric) + epsilon)
    return K_normal, K_abnormal
    
def update_loss_weights_TP_rate(old_weights, TN, TP, FN, FP, epsilon=0.1, alpha=0.5):
    '''
    High alpha => High importance to old weights
    Low alpha => High importance to new weights
    '''
    K_abnormal_new = FN / ((TP+epsilon)*0.9)
    K_normal_new = FP / ((TN+epsilon)*0.1)

    K_abnormal_old = old_weights[1]
    K_normal_old = old_weights[0]

    K_normal = alpha*K_normal_old + (1-alpha)*K_normal_new
    K_abnormal = alpha*K_abnormal_old + (1-alpha)*K_abnormal_new
    
    return K_normal, K_abnormal

# DEC_ADALIGHTLOG ALGORITHM DEFINITION

def DecAdaFed(data):
    """
    Description:
      Implements the FedAvg algorithm, in which for each communication round:
      * all the local servers are trained
      * Central server collects weights of trained models
      * Central server computes the avg_weight given the weight of the trained local models
      * Central server updates all the weights -> avg_weight for next clients training
    Input parameters:
    - update_rule: string expressing how the loss weights will be updated
    - input_size: List of dictionaries containing state_dicts of client models.
    - x_train, y_train, x_valid, y_valid, x_test, y_test: train valid and test datasets
    - num_clients: number of local servers
    - communication_rounds: epochs for the central server model training
    - local_epochs: epochs for the local servers model training in each communication round
    - learning_rate, batch_size: parameters for local servers training
    - output_dir: path in which the trained model and the test metrics will be saved
    - device: cpu, cuda, mps
    Output:
    - model: trained central model
    - test_losses: vector of test losses for each communication round
    - test_accuracies: vector of test accuracies for each communication round
    - test_precisions: vector of test precisions for each communication round
    - test_recalls: vector of test recalls for each communication round
    - test_f1_scores: vector of test F1-scores for each communication round
    """

    check_alpha = data.get("check_alpha")
    update_rule = data.get("update_rule")
    score = data.get("score")
    input_size = data.get("input_size")
    x_train = data.get("x_train")
    y_train = data.get("y_train")
    x_valid = data.get("x_valid")
    y_valid = data.get("y_valid")
    x_test = data.get("x_test")
    y_test= data.get("y_test")
    x_valid_glob= data.get("x_valid_glob")
    y_valid_glob= data.get("y_valid_glob")
    num_clients = data.get("K")
    communication_rounds = data.get("communication_rounds")
    local_epochs = data.get("local_epochs")
    learning_rate = data.get("learning_rate")
    batch_size = data.get("batch_size")
    output_dir = data.get("tcn_models_dir")
    consensus_params = data.get("consensus_params")
    device = data.get("device")

    # Create model folder
    model_path = f"DecAdaLightLog_score={score}_update={update_rule}_lr={learning_rate}_epochs={communication_rounds}"
    tcn_model = output_dir + model_path
    if not os.path.exists(tcn_model+"_00/"):
        tcn_model +="_00/"
        os.makedirs(tcn_model)
    else:
        counter = 1
        while True:
            i=str(counter)
            if counter<10:
                i = "_0"+i+"/"
            else:
                i = "_"+i+"/"
            output_dir = tcn_model +i
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                tcn_model = output_dir
                break
            else:
                counter +=1         
    # Check the update rule
    updates = [
        'f1',
        'recall',
        'TP_rate',
        'precision',
        'no', # case of no weight updates
    ]
    possible_scores = [
        'f1',
        'recall',
        'precision',
        'accuracy',
        'no'
    ]
    if update_rule not in updates:
        raise ValueError(f'Update rule {update_rule} is not recognized.')
    if score not in possible_scores:
        raise ValueError(f'Score {update_rule} for weight averaging is not recognized.')
        
    # Initialize loss weights
    K_normal = 1
    K_abnormal = 1
    weight = torch.tensor([K_normal, K_abnormal],dtype=torch.float).to(device)
    
    # Define local model, optimizer and loss function for each local server
    models = []
    optimizers = []
    schedulers = []
    criteria = []
    #for i in range(0, num_clients +1):
    for i in range(0, num_clients):
        models.append(TCN_model(input_size, 2).to(device))
        if i!=0:
            models[i].load_state_dict(models[0].state_dict())
        criteria.append(nn.CrossEntropyLoss(weight=weight))
        optimizers.append(optim.Adam(models[i].parameters(), lr=learning_rate))
        schedulers.append(lr_scheduler.StepLR(optimizers[i], step_size=20, gamma=0.98))

    # Vectors that will contain the output metrics
    global_losses = []
    global_accuracies = []
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []
    
    # Federated training
    for epoch in range(communication_rounds):
        
        start = time.perf_counter()
        print(f'* Starting communication round {epoch+1}:')
        if epoch < 9:
            epoch_str = f" 0{epoch+1}/{communication_rounds}"
        else:
            epoch_str = f"{epoch+1}/{communication_rounds}"

        client_losses = []
        client_accuracies = []

        for i in range(num_clients):
            print(f' - Training client {i +1} ', end = '')
            print(f'with local loss weights: {criteria[i].weight[0].item():.4f}, {criteria[i].weight[1].item():.4f}...')
            # Create dataloader for training i-th client
            train_dataset = TensorDataset(x_train[i], y_train[i])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Train local model of i-th client
            '''
            local_model = models[i]
            local_optimizer = optimizers[i]
            local_scheduler = schedulers[i]
            local_model = train_local_model(
                local_model,\
                train_loader,\
                criterion,\
                local_optimizer,\
                local_scheduler,\
                local_epochs
            )
            '''
            models[i] = train_local_model(
                models[i],\
                train_loader,\
                criteria[i],\
                optimizers[i],\
                schedulers[i],\
                local_epochs
            )
            
            # Calculate loss on trained i-th local model
            client_loss = calculate_loss(models[i], train_loader, criteria[i])
            client_losses.append(client_loss)
            '''
            #### fare update pesi del criterio i-esimo
            valid_dataset = TensorDataset(x_valid[i], y_valid[i])
            valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
            val_loss, val_accuracy, val_precision, val_recall, val_f1, TP, FP, TN, FN = get_validation_metrics(
                models[i],\
                valid_loader,\
                criteria[i],\
                device
            )
            print(f'   ⮑ Train Loss: {client_loss:.4f}, Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%, Prec: {val_precision:.2f}%, Rec: {val_recall:.2f}%, F1: {val_f1:.2f}%')
            #client_weights.append(local_model.state_dict()) MODIFICATO DA EDS

            # Update local weights (& criterion) for each client
            if update_rule == 'f1':
                K_0, K_1 = update_loss_weights_f1_score(val_f1/100)
            elif update_rule == 'recall':
                K_0, K_1 = update_loss_weights_recall(val_recall/100)
            elif update_rule == 'TP_rate':
                print(f'   ⮑ TP: {TP} | FN: {FN} | TN: {TN} | FP: {FP}')
                weights_to_update = []
                weights_to_update.append(criteria[i].weight[0].item())
                weights_to_update.append(criteria[i].weight[1].item())
                alpha = 0.2
                K_0, K_1 = update_loss_weights_TP_rate(
                    weights_to_update, TN, TP, FN, FP, alpha=alpha
                )
            elif update_rule == 'precision':
                K_0, K_1 = update_loss_weights_precision(val_precision/100)
            else:
                K_0 = K_normal
                K_1 = K_abnormal
            
            local_weights = torch.tensor([K_0, K_1],dtype=torch.float).to(device)
            criteria[i] = nn.CrossEntropyLoss(weight=local_weights)
            '''
            # Calculate training accuracy
            local_train_accuracy = evaluate_train_accuracy(models[i], train_loader, device)
            client_accuracies.append(local_train_accuracy)

        # (1) WEIGHTED MODEL AVERAGE
        if score!='no':
            scores = evaluate_clients_scores(score, models, num_clients, criteria, x_valid_glob, y_valid_glob, device)
            global_weights = decentralized_weighted_average_model_weights(models, scores, consensus_params, True)
        else:
            #global_weights = average_model_weights(models)
            scores = [1,1,1,1,1]
            global_weights = decentralized_weighted_average_model_weights(models, scores, consensus_params, False)
            #global_weights = decentralized_average_model_weights(models, consensus_params)
        # Update models with new global weights
        #for model in models:
        #    model.load_state_dict(global_weights)
        
        # (2) ADAPTIVE LOSS
        #print(f" - Testing...", end= '')
        # TODO: Test all the num_clients models with their validation datasets
        #valid_dataset = TensorDataset(x_valid[i], y_valid[i])
    
        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=True)
        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_TP, test_FP, test_TN, test_FN = get_validation_metrics(
            models[-1],\
            test_loader,\
            criteria[-1],\
            device
        )
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1_scores.append(test_f1)
        end = time.perf_counter()
        print(f"communication round {epoch+1} ended ({end-start:.4f}s):")
        print(f"   ⮑ Loss:      {test_loss:.4f}")
        print(f"   ⮑ Accuracy:  {test_accuracy:.2f}%")
        print(f"   ⮑ Precision: {test_precision:.2f}%")
        print(f"   ⮑ Recall:    {test_recall:.2f}%")
        print(f"   ⮑ F1 score:  {test_f1:.2f}%")
        print(f"   ⮑ TP: {test_TP} | FN: {test_FN} | TN: {test_TN} | FP: {test_FP}")
        if update_rule == 'f1':
            #K_0, K_1 = update_loss_weights_f1_score(test_f1/100)
            K_0, K_1= update_loss_weights(test_f1/100)
        elif update_rule == 'recall':
            K_0, K_1= update_loss_weights(test_recall/100)
            #K_0, K_1 = update_loss_weights_recall(test_recall/100)

        elif update_rule == 'precision':
            K_0, K_1= update_loss_weights(test_precision/100)
            #K_0, K_1 = update_loss_weights_precision(test_precision/100)
        else:
            K_0 = K_normal
            K_1 = K_abnormal
        local_weights = torch.tensor([K_0, K_1],dtype=torch.float).to(device)
        for i in range(num_clients):
            criteria[i] = nn.CrossEntropyLoss(weight=local_weights)

    # Save in model folder the model and all the variables
    model = tcn_model + "model.pth"
    metrics = {
        'loss': test_losses,
        'accuracy': test_accuracies,
        'precision': test_precisions,
        'recall': test_recalls,
        'f1': test_f1_scores
    }
    metrics_dir = tcn_model + "metrics.json"
    with open(metrics_dir, 'w') as json_file:
        json.dump(metrics, json_file)
    torch.save(models[-1].state_dict(), model)
    
    return models[-1], test_losses, test_accuracies, test_precisions, test_recalls, test_f1_scores

def DecAdaFed(data):
    """
    Description:
      Implements the FedAvg algorithm, in which for each communication round:
      * all the local servers are trained
      * Central server collects weights of trained models
      * Central server computes the avg_weight given the weight of the trained local models
      * Central server updates all the weights -> avg_weight for next clients training
    Input parameters:
    - update_rule: string expressing how the loss weights will be updated
    - input_size: List of dictionaries containing state_dicts of client models.
    - x_train, y_train, x_valid, y_valid, x_test, y_test: train valid and test datasets
    - num_clients: number of local servers
    - communication_rounds: epochs for the central server model training
    - local_epochs: epochs for the local servers model training in each communication round
    - learning_rate, batch_size: parameters for local servers training
    - output_dir: path in which the trained model and the test metrics will be saved
    - device: cpu, cuda, mps
    Output:
    - model: trained central model
    - test_losses: vector of test losses for each communication round
    - test_accuracies: vector of test accuracies for each communication round
    - test_precisions: vector of test precisions for each communication round
    - test_recalls: vector of test recalls for each communication round
    - test_f1_scores: vector of test F1-scores for each communication round
    """

    check_alpha = data.get("check_alpha")
    update_rule = data.get("update_rule")
    score = data.get("score")
    input_size = data.get("input_size")
    x_train = data.get("x_train")
    y_train = data.get("y_train")
    x_valid = data.get("x_valid")
    y_valid = data.get("y_valid")
    x_test = data.get("x_test")
    y_test= data.get("y_test")
    x_valid_glob= data.get("x_valid_glob")
    y_valid_glob= data.get("y_valid_glob")
    num_clients = data.get("K")
    communication_rounds = data.get("communication_rounds")
    local_epochs = data.get("local_epochs")
    learning_rate = data.get("learning_rate")
    batch_size = data.get("batch_size")
    output_dir = data.get("tcn_models_dir")
    consensus_params = data.get("consensus_params")
    decentralized = data.get("decentralized")
    device = data.get("device")

    # Create model folder
    if decentralized == True:
        output_dir += 'DecAdaLightLog/'
    else:
        output_dir += 'AdaLightLog/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = f"score={score}_update={update_rule}_lr={learning_rate}_epochs={communication_rounds}"
    tcn_model = output_dir + model_path
    if not os.path.exists(tcn_model+"_00/"):
        tcn_model +="_00/"
        os.makedirs(tcn_model)
    else:
        counter = 1
        while True:
            i=str(counter)
            if counter<10:
                i = "_0"+i+"/"
            else:
                i = "_"+i+"/"
            output_dir = tcn_model +i
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                tcn_model = output_dir
                break
            else:
                counter +=1         
    # Check the update rule
    updates = [
        'f1',
        'recall',
        'TP_rate',
        'precision',
        'no', # case of no weight updates
    ]
    possible_scores = [
        'f1',
        'recall',
        'precision',
        'accuracy',
        'no'
    ]
    if update_rule not in updates:
        raise ValueError(f'Update rule {update_rule} is not recognized.')
    if score not in possible_scores:
        raise ValueError(f'Score {update_rule} for weight averaging is not recognized.')
        
    # Initialize loss weights
    K_normal = 1
    K_abnormal = 1
    weight = torch.tensor([K_normal, K_abnormal],dtype=torch.float).to(device)
    
    # Define local model, optimizer and loss function for each local server
    models = []
    optimizers = []
    schedulers = []
    criteria = []
    #for i in range(0, num_clients +1):
    for i in range(0, num_clients):
        models.append(TCN_model(input_size, 2).to(device))
        if i!=0:
            models[i].load_state_dict(models[0].state_dict())
        criteria.append(nn.CrossEntropyLoss(weight=weight))
        optimizers.append(optim.Adam(models[i].parameters(), lr=learning_rate))
        schedulers.append(lr_scheduler.StepLR(optimizers[i], step_size=20, gamma=0.98))

    # Vectors that will contain the output metrics
    global_losses = []
    global_accuracies = []
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1_scores = []
    
    # Federated training
    for epoch in range(communication_rounds):
        
        start = time.perf_counter()
        print(f'* Starting communication round {epoch+1}:')
        if epoch < 9:
            epoch_str = f" 0{epoch+1}/{communication_rounds}"
        else:
            epoch_str = f"{epoch+1}/{communication_rounds}"

        client_losses = []
        client_accuracies = []

        for i in range(num_clients):
            print(f' - Training client {i +1} ', end = '')
            print(f'with local loss weights: {criteria[i].weight[0].item():.4f}, {criteria[i].weight[1].item():.4f}...')
            # Create dataloader for training i-th client
            train_dataset = TensorDataset(x_train[i], y_train[i])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Train local model of i-th client
            models[i] = train_local_model(
                models[i],\
                train_loader,\
                criteria[i],\
                optimizers[i],\
                schedulers[i],\
                local_epochs
            )
            
            # Calculate loss on trained i-th local model
            client_loss = calculate_loss(models[i], train_loader, criteria[i])
            client_losses.append(client_loss)
            
        
        # (1) WEIGHTED MODEL AVERAGE
        ## è una procedura che va fatta su un dataset comune!!!
        if decentralized:
            # DECADALIGHTLOG
            # media pesata dei parametri implica l'esistenza di un dataset comune. 
            # In un contesto decentralizzato questo non puo avvenire: DecAdaLightLog non fa uso di weighted model average
            global_weights = decentralized_average_model_weights(models, consensus_params)
        else:
            # ADALIGHTLOG (CENTRALIZED FEDERATED LEARNING)
            if score != 'no':
                scores = evaluate_clients_scores(score, models, num_clients, criteria, x_valid_glob, y_valid_glob, device)
                global_weights = average_weighted_model_weights(models,scores)
            else:
                global_weights = average_model_weights(models)
        '''
        if score!='no':
            scores = evaluate_clients_scores(score, models, num_clients, criteria, x_valid_glob, y_valid_glob, device)
            if decentralized==True:
                global_weights = decentralized_weighted_average_model_weights(models, scores, consensus_params, True)
            else:
                global_weights = average_weighted_model_weights(models,scores)
        else:
            #global_weights = average_model_weights(models)
            if decentralized==True:
                scores = [1,1,1,1,1]
                global_weights = decentralized_weighted_average_model_weights(models, scores, consensus_params, False)
            else:
                global_weights = average_model_weights(models)
        '''
        # (2) ADAPTIVE LOSS FOR CLIENT i-TH
        print(f' - Updating clients\' losses, using local validation datasets:')
        for i in range(num_clients):
            valid_dataset = TensorDataset(x_valid[i], y_valid[i])
            valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
            val_loss, val_accuracy, val_precision, val_recall, val_f1, TP, FP, TN, FN = get_validation_metrics(
                models[i],\
                valid_loader,\
                criteria[i],\
                device
            )
            print(f'   ⮑ Client {i+1}: Val Loss: {val_loss:.4f}, Acc: {val_accuracy:.2f}%, Prec: {val_precision:.2f}%, Rec: {val_recall:.2f}%, F1: {val_f1:.2f}%')

            # Update local weights (& criterion) for each client
            if update_rule == 'f1':
                #K_0, K_1 = update_loss_weights_f1_score(test_f1/100)
                K_0, K_1= update_loss_weights(val_f1/100)
            elif update_rule == 'recall':
                K_0, K_1= update_loss_weights(val_recall/100)
                #K_0, K_1 = update_loss_weights_recall(test_recall/100)
    
            elif update_rule == 'precision':
                K_0, K_1= update_loss_weights(val_precision/100)
                #K_0, K_1 = update_loss_weights_precision(test_precision/100)
            else:
                K_0 = K_normal
                K_1 = K_abnormal
            local_weights = torch.tensor([K_0, K_1],dtype=torch.float).to(device)
            criteria[i] = nn.CrossEntropyLoss(weight=local_weights)

            if i == num_clients-1:
                
                test_losses.append(val_loss)
                test_accuracies.append(val_accuracy)
                test_precisions.append(val_precision)
                test_recalls.append(val_recall)
                test_f1_scores.append(val_f1)
                end = time.perf_counter()
                print(f"communication round {epoch+1} ended ({end-start:.4f}s):")
                print(f"   ⮑ Loss:      {val_loss:.4f}")
                print(f"   ⮑ Accuracy:  {val_accuracy:.2f}%")
                print(f"   ⮑ Precision: {val_precision:.2f}%")
                print(f"   ⮑ Recall:    {val_recall:.2f}%")
                print(f"   ⮑ F1 score:  {val_f1:.2f}%")
                print(f"   ⮑ TP: {TP} | FN: {FN} | TN: {TN} | FP: {FP}")

    # Save in model folder the model and all the variables
    model = tcn_model + "model.pth"
    metrics = {
        'loss': test_losses,
        'accuracy': test_accuracies,
        'precision': test_precisions,
        'recall': test_recalls,
        'f1': test_f1_scores
    }
    metrics_dir = tcn_model + "metrics.json"
    with open(metrics_dir, 'w') as json_file:
        json.dump(metrics, json_file)
    torch.save(models[-1].state_dict(), model)
    
    return models[-1], test_losses, test_accuracies, test_precisions, test_recalls, test_f1_scores
    

