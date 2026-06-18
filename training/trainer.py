import torch
import torch.nn as nn
from data.encoding import get_encoder

def train_epoch(model, loader, optimizer, criterion, encoder, num_steps, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        images = encoder(images)
        optimizer.zero_grad()
        outputs = model(images, num_steps)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(loader)
    return avg_loss

def evaluate(model, loader, criterion, encoder, num_steps, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            images = encoder(images)
            outputs = model(images, num_steps)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            answer = torch.argmax(outputs, dim=1)
            correct += (answer == labels).sum().item()
            total += labels.size(0)
        accuracy = correct / total
        avg_loss = total_loss / len(loader)
    return avg_loss, accuracy