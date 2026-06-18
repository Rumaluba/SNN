import torch
import torch.nn as nn
from model.lif_neuron import LIFNeuron

class SNN(nn.Module):
    def __init__(self, beta, threshold, spike_grad, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.lif1  = LIFNeuron(beta, threshold, spike_grad)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(128)
        self.lif2  = LIFNeuron(beta, threshold, spike_grad)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(256)
        self.lif3  = LIFNeuron(beta, threshold, spike_grad)
        self.pool3 = nn.MaxPool2d(2)

        self.flatten = nn.Flatten()

        self.fc1  = nn.Linear(256 * 8 * 8, 512)
        self.lif4 = LIFNeuron(beta, threshold, spike_grad)
        self.fc2  = nn.Linear(512, num_classes)

    def reset_state(self):
        for m in self.modules():
            if isinstance(m, LIFNeuron):
                m.reset_state()

    def forward(self, x, num_steps):
        self.reset_state()
        spike_record = []
        for t in range(num_steps):
            spk1 = self.lif1(self.bn1(self.conv1(x[t])))
            spk2 = self.pool2(self.lif2(self.bn2(self.conv2(spk1))))
            spk3 = self.pool3(self.lif3(self.bn3(self.conv3(spk2))))
            spk4 = self.lif4(self.fc1(self.flatten(spk3)))
            out = self.fc2(spk4)
            spike_record.append(out)
        return torch.stack(spike_record, dim=0).mean(dim=0)