import torch
import torch.nn as nn
from data.attacks import fgsm, pgd, gaussian_noise


def evaluate_robustness(model, test_loader, encoder, num_steps, device,
                        epsilon=0.03, pgd_alpha=0.007, pgd_steps=10,
                        noise_levels=[0.05, 0.1, 0.2, 0.3, 0.5], m=5):
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

    results['clean'] = get_accuracy(test_loader)

    model.eval()
    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv = fgsm(model, images, labels, epsilon, criterion, encoder, num_steps, m=m)
        encoded = encoder(adv)
        with torch.no_grad():
            outputs = model(encoded, num_steps)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    results['fgsm'] = correct / total

    correct, total = 0, 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        adv = pgd(model, images, labels, epsilon, pgd_alpha, pgd_steps,
                  criterion, encoder, num_steps, m=m)
        encoded = encoder(adv)
        with torch.no_grad():
            outputs = model(encoded, num_steps)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    results['pgd'] = correct / total

    for noise_level in noise_levels:
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
        results[f'gaussian_{noise_level}'] = correct / total

    return results