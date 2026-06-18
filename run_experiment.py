import torch
import torch.optim as optim
import torch.nn as nn
import os
from model.builder import get_model
from data.encoding import get_encoder
from data.dataset import get_dataloaders, get_mnist_dataloaders
from training.metrics import (plot_loss_accuracy, plot_confusion_matrix,
                              plot_class_accuracy, plot_robustness)
from training.trainer import train_epoch, evaluate
from training.robustness_eval import evaluate_robustness


def run_experiment(experiment_name, beta, threshold, spike_grad_name,
                   num_steps, encoding, arch='vgg11', dataset='cifar10',
                   epochs=200, batch_size=128, lr=0.001, device='cuda'):
    save_dir = f"results/{experiment_name}"
    os.makedirs(save_dir, exist_ok=True)

    if dataset == 'mnist':
        train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size)
        in_channels = 1
    else:
        train_loader, val_loader, test_loader = get_dataloaders(batch_size)
        in_channels = 3

    encoder = get_encoder(encoding, num_steps)
    model = get_model(arch, in_channels, 10, beta, threshold, spike_grad_name)
    model = model.to(device)

    model_path = f"{save_dir}/model.pth"
    if os.path.exists(model_path):
        print(f"Model found for {experiment_name}, skipping training")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        train_losses, val_losses, val_accuracies = [], [], []
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                     encoder, num_steps, device)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion,
                                              encoder, num_steps, device)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            scheduler.step()
            print(f"Epoch {epoch + 1}/{epochs} | train_loss: {train_loss:.4f} | "
                  f"val_loss: {val_loss:.4f} | val_acc: {val_accuracy:.4f}")
        plot_loss_accuracy(train_losses, val_losses, val_accuracies,
                           f"{save_dir}/loss_accuracy.png")
        torch.save(model.state_dict(), model_path)

    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss, test_accuracy = evaluate(model, test_loader, criterion,
                                        encoder, num_steps, device)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            images = encoder(images)
            outputs = model(images, num_steps)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    robustness_results = evaluate_robustness(model, test_loader, encoder, num_steps, device)

    plot_confusion_matrix(all_labels, all_preds, f"{save_dir}/confusion_matrix.png")
    plot_class_accuracy(all_labels, all_preds, f"{save_dir}/class_accuracy.png")
    plot_robustness(robustness_results, experiment_name, f"{save_dir}/robustness.png")

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Robustness: {robustness_results}")
    return test_accuracy, robustness_results
