import torch
from data.attacks import fgsm, pgd, gaussian_noise
import torch.nn as nn


def evaluate_robustness(model, test_loader, encoder, num_steps, device,
                        epsilon=0.03, pgd_alpha=0.007, pgd_steps=10,
                        noise_level=0.1):
    criterion = nn.CrossEntropyLoss()
    results = {}

    def get_accuracy(loader_or_list):
        correct, total = 0, 0
        for images, labels in loader_or_list:
            images, labels = images.to(device), labels.to(device)
            encoded = encoder(images)
            with torch.no_grad():
                outputs = model(encoded, num_steps)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / total

    # clean accuracy
    results['clean'] = get_accuracy(test_loader)

    # FGSM
    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv = fgsm(model, images, labels, epsilon, criterion, encoder, num_steps)
        encoded = encoder(adv)
        with torch.no_grad():
            outputs = model(encoded, num_steps)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    results['fgsm'] = correct / total

    # PGD
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv = pgd(model, images, labels, epsilon, pgd_alpha, pgd_steps,
                  criterion, encoder, num_steps)
        encoded = encoder(adv)
        with torch.no_grad():
            outputs = model(encoded, num_steps)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    results['pgd'] = correct / total

    # Gaussian noise
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        noisy = gaussian_noise(images, noise_level)
        encoded = encoder(noisy)
        with torch.no_grad():
            outputs = model(encoded, num_steps)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    results['gaussian'] = correct / total

    return results