import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


CIFAR10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

def plot_loss_accuracy(train_losses, val_losses, val_accuracies, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, label='train')
    ax1.plot(val_losses, label='val')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(val_accuracies, label='val', color='green')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CIFAR10_classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_class_accuracy(y_true, y_pred, save_path):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracies = []
    for i in range(10):
        mask = y_true == i
        acc = (y_pred[mask] == y_true[mask]).mean()
        accuracies.append(acc)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(CIFAR10_classes, accuracies, color='steelblue')
    ax.set_title('Accuracy per class')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_sweep_results(param_values, accuracies, param_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([str(v) for v in param_values], accuracies, color='steelblue')
    ax.set_title(f'Accuracy vs {param_name}')
    ax.set_xlabel(param_name)
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(0, 1)
    for i, acc in enumerate(accuracies):
        ax.text(i, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_robustness(results, experiment_name, save_path):
    labels = list(results.keys())
    values = list(results.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=['steelblue', 'tomato', 'orange', 'gray'])
    ax.set_title(f'Robustness — {experiment_name}')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()