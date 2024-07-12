"""
Contains utility functions for the project.
"""
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.utils
import torch.utils.data
import numpy as np

import json
import os

def set_seeds(seed: int) -> None:
    """Sets seeds for reproducibility.

    Args:
    seed: A seed to set for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_curve(results: dict, arch_name: str):
    """Save training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}
        arch_name (str): name of the architecture
    """
    train_loss = np.array( results["train_loss"] )
    val_loss = np.array( results["val_loss"] )
    train_acc = np.array( results["train_acc"] )
    val_acc = np.array( results["val_acc"] )

    # Membuat subplots untuk train dan validation
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
                        'Loss', 'Accuracy'],
                        horizontal_spacing=0.05, )

    # Membuat plot untuk train loss dan accuracy (subplot kiri)
    fig.add_trace(go.Scatter(y=train_loss, mode='lines', line=dict(
        color='blue'), name='Original'), row=1, col=1)
    fig.add_trace(go.Scatter(y=val_loss, mode='lines', line=dict(
        color='orange'), name='CLAHE'), row=1, col=1)

    # Membuat plot untuk validation loss dan accuracy (subplot kanan)
    fig.add_trace(go.Scatter(y=train_acc, mode='lines', line=dict(
        color='blue'), name=''), row=1, col=2)
    fig.add_trace(go.Scatter(y=val_acc, mode='lines', line=dict(
        color='orange'), name=''), row=1, col=2)

    # Mengatur layout dan judul plot
    fig.update_layout(
        xaxis1_title='Epoch',
        xaxis2_title='Epoch',
        yaxis1_title='Value',
        yaxis2_title='Value',
        width=1800,  # Lebar plot dalam piksel
        height=700,  # Tinggi plot dalam piksel
        font=dict(size=15),  # Ukuran font
    )

    fig.write_image(f"./results/{arch_name}/plots_loss_acc_curve.png")

def evaluate_model(model: torch.nn.Module, 
                   dataloader: torch.utils.data.DataLoader, 
                   arch_name: str,
                   class_names: list,
                   device: torch.device):
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred_logits = model(X)
            y_pred_labels = y_pred_logits.argmax(dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(y_pred_labels.cpu().numpy())
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro")
    sensitivity = recall_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print(f"Evaluation {arch_name} results:")
    print(f"Accuracy: {acc}")
    print(f"Precision: {precision}")
    print(f"Sensitivity: {sensitivity}")
    print(f"Specificity: {specificity}")

    # save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.savefig(f"./results/{arch_name}/confusion_matrix.png")

    # save evaluatuon metrics to csv
    eval_metrics = {
        "Architecture": arch_name,
        "Accuracy": acc,
        "Precision": precision,
        "Sensitivity": sensitivity,
        "Specificity": specificity
    }
    df = pd.DataFrame([eval_metrics])
    df.to_csv(f"./results/{arch_name}/evaluation_metrics.csv", index=False)

def save_to_json(data:dict, file_name:str):
    if not os.path.isfile(file_name):
        with open(file_name, "w") as file:
            json.dump(data, file)