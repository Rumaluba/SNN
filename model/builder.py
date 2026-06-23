import torch
import torch.nn as nn
from model.lif_neuron import LIFNeuron

CONFIGS = {
    'vgg11': [
        [64, 'A'],
        [128, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512],
        []
    ]
}

class VGGSNN(nn.Module):
    def __init__(self, arch_name, in_channels, num_classes, beta, threshold, spike_grad_name):
        super().__init__()
        cfg = CONFIGS[arch_name]
        self.features = make_layers(cfg, in_channels, beta, threshold, spike_grad_name)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def reset_state(self):
        for m in self.modules():
            if isinstance(m, LIFNeuron):
                m.reset_state()

    def forward(self, x, num_steps):
        self.reset_state()
        spike_record = []
        for t in range(num_steps):
            out = self.features(x[t])
            out = self.pool(out).flatten(1)
            out = self.fc(out)
            spike_record.append(out)
        return torch.stack(spike_record, dim=0).mean(dim=0)


def make_layers(cfg, in_channels, beta, threshold, spike_grad_name):
    layers = []
    for block in cfg:
        for v in block:
            if v == 'A':
                layers.append(nn.AvgPool2d(2))
            else:
                layers += [
                    nn.Conv2d(in_channels, v, 3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    LIFNeuron(beta=beta, threshold=threshold, spike_grad_name=spike_grad_name)
                ]
                in_channels = v
    return nn.Sequential(*layers)

def get_model(name, in_channels, num_classes, beta, threshold, spike_grad_name):
    return VGGSNN(name, in_channels, num_classes, beta, threshold, spike_grad_name)