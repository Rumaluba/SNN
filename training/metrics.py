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

    colors = []
    for label in labels:
        if label == 'clean':
            colors.append('steelblue')
        elif label == 'fgsm':
            colors.append('tomato')
        elif label == 'pgd':
            colors.append('orange')
        else:
            colors.append('gray')

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_title(f'Robustness - {experiment_name}')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_beta_summary(betas, results_by_T, save_path):
    fig, ax = plt.subplots(figsize=(12, 7))

    line_styles = {4: '-', 8: '--'}
    markers = {4: 'o', 8: 's'}
    colors = {
        'clean': 'tab:blue',
        'fgsm': 'tab:red',
        'pgd': 'tab:orange',
    }
    gray_shades = ['#888888', '#aaaaaa', '#666666', '#444444', '#bbbbbb']

    for T, results_list in results_by_T.items():
        if len(results_list) == 0:
            continue

        n = len(results_list)
        b = betas[:n]

        # clean / fgsm / pgd
        for key in ['clean', 'fgsm', 'pgd']:
            values = [r[key] for r in results_list]
            ax.plot(b, values, marker=markers.get(T, 'o'),
                    color=colors[key], linestyle=line_styles.get(T, '-'),
                    label=f'{key.upper()} (T={T})')

        # все gaussian_* ключи
        gaussian_keys = sorted([k for k in results_list[0].keys() if k.startswith('gaussian_')])
        for idx, gkey in enumerate(gaussian_keys):
            values = [r[gkey] for r in results_list]
            sigma = gkey.split('_')[1]
            ax.plot(b, values, marker=markers.get(T, 'o'),
                    color=gray_shades[idx % len(gray_shades)],
                    linestyle=line_styles.get(T, '-'),
                    label=f'Gaussian ?={sigma} (T={T})')

    ax.set_xlabel('Beta')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy/Beta')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()