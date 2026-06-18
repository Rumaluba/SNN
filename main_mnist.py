import torch
import random
import numpy as np
from run_experiment import run_experiment
from training.metrics import plot_sweep_results

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

device = "cuda:0"
print(f"Using device: {device}")

betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
timesteps = [4, 8, 16]
encoding = "rate"
spike_grad_name = "fast_sigmoid"
threshold = 1.0
epochs = 200
batch_size = 128
lr = 0.001
dataset = "mnist"

all_accuracies = {}
all_robustness = {}

for num_steps in timesteps:
    beta_accs = []
    beta_robust = []

    for beta in betas:
        exp_name = f"{dataset}_vgg11_beta{beta}_T{num_steps}"
        print(f"\nRunning: {exp_name}")

        acc, robustness = run_experiment(
            experiment_name=exp_name,
            beta=beta,
            threshold=threshold,
            spike_grad_name=spike_grad_name,
            num_steps=num_steps,
            encoding=encoding,
            arch='vgg11',
            dataset=dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            device=device
        )
        beta_accs.append(acc)
        beta_robust.append(robustness)

    all_accuracies[num_steps] = beta_accs
    all_robustness[num_steps] = beta_robust

    plot_sweep_results(
        betas, beta_accs,
        f"beta (T={num_steps})",
        f"results/{dataset}_sweep_beta_T{num_steps}.png"
    )

for i, beta in enumerate(betas):
    accs_per_T = [all_accuracies[T][i] for T in timesteps]
    plot_sweep_results(
        timesteps, accs_per_T,
        f"timesteps (beta={beta})",
        f"results/{dataset}_sweep_T_beta{beta}.png"
    )

for num_steps in timesteps:
    for attack in ['clean', 'fgsm', 'pgd', 'gaussian']:
        attack_accs = [all_robustness[num_steps][i][attack] for i in range(len(betas))]
        plot_sweep_results(
            betas, attack_accs,
            f"beta — {attack} (T={num_steps})",
            f"results/{dataset}_robustness_{attack}_T{num_steps}.png"
        )

print("\nMNIST experiments done!")