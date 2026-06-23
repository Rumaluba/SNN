import torch
import torch.nn as nn


def fgsm(model, images, labels, epsilon, criterion, encoder, num_steps, m=5):
    images = images.clone().detach().requires_grad_(True)
    outputs = None
    for _ in range(m):
        encoded = encoder(images)
        out = model(encoded, num_steps)
        outputs = out if outputs is None else outputs + out
    outputs = outputs / m
    loss = criterion(outputs, labels)
    loss.backward()
    adv_images = images + epsilon * images.grad.sign()
    return adv_images.clamp(0, 1).detach()


def pgd(model, images, labels, epsilon, alpha, num_iter, criterion, encoder, num_steps, m=5):
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = adv_images.clamp(0, 1)

    for _ in range(num_iter):
        adv_images = adv_images.clone().detach().requires_grad_(True)
        outputs = None
        for _ in range(m):
            encoded = encoder(adv_images)
            out = model(encoded, num_steps)
            outputs = out if outputs is None else outputs + out
        outputs = outputs / m
        loss = criterion(outputs, labels)
        loss.backward()
        adv_images = adv_images + alpha * adv_images.grad.sign()
        delta = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = (images + delta).clamp(0, 1).detach()

    return adv_images


def gaussian_noise(images, noise_level):
    noise = torch.randn_like(images) * noise_level
    return (images + noise).clamp(0, 1)