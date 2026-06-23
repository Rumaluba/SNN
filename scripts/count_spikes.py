import os
import torch
import gc
from data.dataset import get_dataloaders
from data.encoding import get_encoder
from model.builder import get_model
from model.lif_neuron import LIFNeuron
import matplotlib.pyplot as plt


def compute_spike_rate(model, test_loader, encoder, num_steps, device):
    model.eval()
    total_spikes = 0.0
    total_neurons = 0
    num_batches = 0

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            encoded = encoder(images)
            model.reset_state()

            for t in range(num_steps):
                _ = model.features(encoded[t])
                for m in model.modules():
                    if isinstance(m, LIFNeuron) and hasattr(m, 'last_spike'):
                        total_spikes += m.last_spike.sum().item()
                        if t == 0 and num_batches == 0:
                            total_neurons += m.last_spike.numel()

            num_batches += 1

    avg_spike_rate = total_spikes / (num_batches * num_steps * total_neurons) if total_neurons > 0 else 0.0
    return avg_spike_rate


device = "cuda" if torch.cuda.is_available() else "cpu"
betas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
timesteps = [4, 8]
threshold = 1.0
spike_grad_name = "fast_sigmoid"
batch_size = 128
dataset = "cifar10"

for encoding in ["rate", "const"]:
    _, _, test_loader = get_dataloaders(batch_size)
    results = {}

    for num_steps in timesteps:
        spike_rates = []
        for beta in betas:
            exp_name = f"{dataset}_{encoding}_vgg11_beta{beta}_T{num_steps}"
            model_path = f"results/{exp_name}/model.pth"

            if not os.path.exists(model_path):
                print(f"Skipping {exp_name} - no model found")
                spike_rates.append(None)
                continue

            model = get_model('vgg11', 3, 10, beta, threshold, spike_grad_name)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)

            encoder = get_encoder(encoding, num_steps)
            sr = compute_spike_rate(model, test_loader, encoder, num_steps, device)
            spike_rates.append(sr)
            print(f"{exp_name}: spike_rate = {sr:.4f}")

            del model, encoder
            torch.cuda.empty_cache()
            gc.collect()

        results[num_steps] = spike_rates

    fig, ax = plt.subplots(figsize=(10, 5))
    for num_steps in timesteps:
        valid_betas = [betas[i] for i in range(len(betas)) if results[num_steps][i] is not None]
        valid_rates = [r for r in results[num_steps] if r is not None]
        ax.plot(valid_betas, valid_rates, marker='o' if num_steps == 4 else 's',
                linestyle='-' if num_steps == 4 else '--',
                label=f'T={num_steps}')

    ax.set_xlabel('Beta')
    ax.set_ylabel('Average Spike Rate')
    ax.set_title(f'Spike Rate vs Beta - {encoding} encoding')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_{encoding}_spike_rate.png', dpi=150)
    plt.close()
    print(f"Saved spike rate plot for {encoding}")

print("Done!")