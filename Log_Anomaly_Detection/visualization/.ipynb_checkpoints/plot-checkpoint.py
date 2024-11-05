import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools 

def plot_metrics(val_losses, val_acc, val_prec, val_rec, val_f1):
    '''
    Plots metrics obtained from the last training 
    '''
    metrics = [
        (val_losses, 'Validation Loss', 'Loss', 'Training Loss over Epochs', 'red'),
        (val_acc, 'Validation Accuracy', 'Accuracy', 'Training Accuracy over Epochs', 'orange'),
        (val_prec, 'Validation Precision', 'Precision', 'Validation Precision over Epochs', 'orange'),
        (val_rec, 'Validation Recall', 'Recall', 'Validation Recall over Epochs', 'orange'),
        (val_f1, 'Validation F1 Score', 'F1', 'Validation F1 Score over Epochs', 'orange')
    ]

    fig, axs = plt.subplots(6, 1, figsize=(10, 30))

    for i, (data, label, ylabel, title, color) in enumerate(metrics):
        axs[i].plot(data, label=label, color=color)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(title)
        axs[i].legend()
        axs[i].grid(True)
    
    # Combined plot for precision, recall, and F1 score
    axs[5].plot(val_prec, label='Validation Precision', color='blue')
    axs[5].plot(val_rec, label='Validation Recall', color='orange')
    axs[5].plot(val_f1, label='Validation F1 Score', color='green')
    axs[5].set_xlabel('Epoch')
    axs[5].set_ylabel('Metrics')
    axs[5].set_title('Validation Metrics over Epochs')
    axs[5].legend()
    axs[5].grid(True)

    plt.tight_layout()
    plt.show()

def plot_loaded_metrics(path):
    '''
    Plots metrics obtained from a given trained model 
    '''
    with open(path, 'r') as json_file:
        data = json.load(json_file)
        
    metrics = {
        "Validation Loss": (np.array(data["loss"]), "Loss", "Training Loss over Epochs", "red"),
        "Validation Accuracy": (np.array(data["accuracy"]), "Accuracy", "Training Accuracy over Epochs", "orange"),
        "Validation Precision": (np.array(data["precision"]), "Precision", "Validation Precision over Epochs", "orange"),
        "Validation Recall": (np.array(data["recall"]), "Recall", "Validation Recall over Epochs", "orange"),
        "Validation F1 Score": (np.array(data["f1"]), "F1", "Validation F1 Score over Epochs", "orange")
    }

    fig, axs = plt.subplots(6, 1, figsize=(10, 30))

    for i, (label, (values, ylabel, title, color)) in enumerate(metrics.items()):
        axs[i].plot(values, label=label, color=color)
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(title)
        axs[i].legend()
        axs[i].grid(True)
    
    # Combined plot for precision, recall, and F1 score
    axs[5].plot(metrics["Validation Precision"][0], label='Validation Precision', color='blue')
    axs[5].plot(metrics["Validation Recall"][0], label='Validation Recall', color='orange')
    axs[5].plot(metrics["Validation F1 Score"][0], label='Validation F1 Score', color='green')
    axs[5].set_xlabel('Epoch')
    axs[5].set_ylabel('Metrics')
    axs[5].set_title('Validation Metrics over Epochs')
    axs[5].legend()
    axs[5].grid(True)

    plt.tight_layout()
    plt.show()

def compare_loaded_metrics(path1, path2, path3):
    '''
    Plots the metrics of 3 models trained with different approached: FedAvg, AdaLightLog, DecAdaLightLog 
    '''
    metrics = ['FedAvg', 'AdaLighLog', 'DecAdaLightLog']
    paths = [path1, path2, path3]
    all_data = []

    # Carica i dati dai file JSON
    for path in paths:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
            all_data.append({
                'loss': np.array(data["loss"]),
                'accuracy': np.array(data["accuracy"]),
                'precision': np.array(data["precision"]),
                'recall': np.array(data["recall"]),
                'f1': np.array(data["f1"])
            })

    fig, axs = plt.subplots(5, 1, figsize=(10, 30))

    titles = ['Training Loss over Epochs', 'Training Accuracy over Epochs', 'Validation Precision over Epochs', 'Validation Recall over Epochs', 'Validation F1 Score over Epochs']
    ylabels = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['green', 'orange', 'blue', 'red']

    for i, (metric, ylabel, title) in enumerate(zip(['loss', 'accuracy', 'precision', 'recall', 'f1'], ylabels, titles)):
        for j in range(len(paths)):
            axs[i].plot(all_data[j][metric], label=f'score {metrics[j]}', color=colors[j])
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(title)
        axs[i].legend()
        axs[i].grid(True)
    
    axs[1].set_ylim(40, 120)  # Specific adjustment for accuracy plot

    plt.tight_layout()
    plt.show()
    
def compare_metrics_fixed_score(path1, path2, path3, path4):
    '''
    Plots the metrics of four different trained models which have the same score metric for averaging model weights, but different loss update rules
    '''
    update_rules = ['f1', 'recall', 'precision','no']

    paths = [path1, path2, path3, path4]
    all_data = []

    for path in paths:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
            all_data.append({
                'loss': np.array(data["loss"]),
                'accuracy': np.array(data["accuracy"]),
                'precision': np.array(data["precision"]),
                'recall': np.array(data["recall"]),
                'f1': np.array(data["f1"])
            })

    fig, axs = plt.subplots(5, 1, figsize=(10, 30))

    titles = ['Training Loss over Epochs', 'Training Accuracy over Epochs', 'Validation Precision over Epochs', 'Validation Recall over Epochs', 'Validation F1 Score over Epochs']
    ylabels = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['green', 'orange', 'blue', 'red']

    for i, (metric, ylabel, title) in enumerate(zip(['loss', 'accuracy', 'precision', 'recall', 'f1'], ylabels, titles)):
        for j in range(4):
            axs[i].plot(all_data[j][metric], label=f'{update_rules[j]} update', color=colors[j])
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(title)
        axs[i].legend()
        axs[i].grid(True)
    
    axs[1].set_ylim(40, 120)  # Specific adjustment for accuracy plot

    plt.tight_layout()
    plt.show()

def compare_metrics_fixed_update(path1, path2, path3, path4):
    '''
    Plots the metrics of four different trained models which have the same loss update rule, but different score metrics for averaging model weights different 
    '''
    metrics = ['f1', 'recall', 'precision', 'accuracy']
    paths = [path1, path2, path3, path4]
    all_data = []

    # Carica i dati dai file JSON
    for path in paths:
        with open(path, 'r') as json_file:
            data = json.load(json_file)
            all_data.append({
                'loss': np.array(data["loss"]),
                'accuracy': np.array(data["accuracy"]),
                'precision': np.array(data["precision"]),
                'recall': np.array(data["recall"]),
                'f1': np.array(data["f1"])
            })

    fig, axs = plt.subplots(5, 1, figsize=(10, 30))

    titles = ['Training Loss over Epochs', 'Training Accuracy over Epochs', 'Validation Precision over Epochs', 'Validation Recall over Epochs', 'Validation F1 Score over Epochs']
    ylabels = ['Loss', 'Accuracy', 'Precision', 'Recall', 'F1']
    colors = ['green', 'orange', 'blue', 'red']

    for i, (metric, ylabel, title) in enumerate(zip(['loss', 'accuracy', 'precision', 'recall', 'f1'], ylabels, titles)):
        for j in range(4):
            axs[i].plot(all_data[j][metric], label=f'score {metrics[j]}', color=colors[j])
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(title)
        axs[i].legend()
        axs[i].grid(True)
    
    axs[1].set_ylim(40, 120)  # Specific adjustment for accuracy plot

    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix(y_true, y_pred, data_dir, classes, normalize=False, title=None, cmap=plt.cm.Blues, fontsize=15, type="AdaLightLog"):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)  # Setting x-axis font size
    plt.yticks(tick_marks, classes, fontsize=15)  # Setting y-axis font size

    fmt = '.4f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=fontsize)  # Setting cell text font size

    plt.ylabel('True label', fontsize=fontsize)  # Setting y-axis label font size
    plt.xlabel('Predicted label', fontsize=fontsize)  # Setting x-axis label font size
    plt.tight_layout()
    plt.savefig(f'{data_dir}CM_{type}.pdf', format="pdf")
    #plt.savefig(data_dir+'/CRITIS_Plots/CM_AdaFed.png')
    plt.show()

