import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import gc
from data.dataset import get_dataloaders
from data.encoding import get_encoder
from model.builder import get_model
from training.trainer import train_epoch, evaluate
from training.metrics import (plot_loss_accuracy, plot_confusion_matrix,
                               plot_class_accuracy, plot_robustness,
                               plot_beta_summary)
from training.robustness_eval import evaluate_robustness


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
timesteps = [4, 8]
encoding = "rate"
spike_grad_name = "fast_sigmoid"
threshold = 1.0
epochs = 200
batch_size = 128
lr = 0.001
dataset = "cifar10"
m = 1 if encoding == "const" else 5

all_robustness = {}

for num_steps in timesteps:
    beta_robust = []

    for beta in betas:
        exp_name = f"{dataset}_{encoding}_vgg11_beta{beta}_T{num_steps}"
        save_dir = f"results/{exp_name}"
        os.makedirs(save_dir, exist_ok=True)
        model_path = f"{save_dir}/model.pth"
        robustness_path = f"{save_dir}/robustness.png"

        # --- TRAINING ---
        train_loader, val_loader, test_loader = get_dataloaders(batch_size)
        encoder = get_encoder(encoding, num_steps)
        model = get_model('vgg11', 3, 10, beta, threshold, spike_grad_name)
        model = model.to(device)

        if os.path.exists(model_path):
            print(f"\nModel found for {exp_name}, loading...")
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            print(f"\nTraining: {exp_name}")
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
                print(f"Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.4f} | "
                      f"val_loss: {val_loss:.4f} | val_acc: {val_accuracy:.4f}")

            plot_loss_accuracy(train_losses, val_losses, val_accuracies,
                               f"{save_dir}/loss_accuracy.png")
            torch.save(model.state_dict(), model_path)

            del optimizer, scheduler, train_losses, val_losses, val_accuracies
            torch.cuda.empty_cache()
            gc.collect()

        confusion_path = f"{save_dir}/confusion_matrix.png"
        if not os.path.exists(confusion_path):
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    enc_images = encoder(images)
                    outputs = model(enc_images, num_steps)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.numpy())
            plot_confusion_matrix(all_labels, all_preds, confusion_path)
            plot_class_accuracy(all_labels, all_preds, f"{save_dir}/class_accuracy.png")

        del train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

        if os.path.exists(robustness_path):
            print(f"Skipping attacks for {exp_name} - already done")
            robustness_results = None
        else:
            print(f"Running attacks: {exp_name}")
            model.eval()
            robustness_results = evaluate_robustness(model, test_loader, encoder,
                                                      num_steps, device, m=m)
            plot_robustness(robustness_results, exp_name, robustness_path)
            print(f"Done: {exp_name} | robustness: {robustness_results}")

        if robustness_results is not None:
            beta_robust.append(robustness_results)

        del model, encoder, test_loader
        torch.cuda.empty_cache()
        gc.collect()

    all_robustness[num_steps] = beta_robust

plot_beta_summary(betas, all_robustness, f"results/{dataset}_{encoding}_beta_summary.png")
print(f"\nAll {encoding} experiments done!")