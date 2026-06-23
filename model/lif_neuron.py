import torch
import torch.nn as nn
import numpy as np


class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, thr):
        ctx.save_for_backward(v - thr)
        return (v >= thr).float()

    @staticmethod
    def backward(ctx, grad):
        diff, = ctx.saved_tensors
        slope = 25.0
        return grad / (1 + slope * diff.abs()) ** 2, None


class ATan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, thr):
        ctx.save_for_backward(v - thr)
        return (v >= thr).float()

    @staticmethod
    def backward(ctx, grad):
        diff, = ctx.saved_tensors
        alpha = 2.0
        return grad * (alpha / 2.0) / (1.0 + (np.pi / 2.0 * alpha * diff) ** 2), None


class Sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, thr):
        ctx.save_for_backward(v - thr)
        return (v >= thr).float()

    @staticmethod
    def backward(ctx, grad):
        diff, = ctx.saved_tensors
        k = 4.0
        sg = torch.sigmoid(diff * k)
        return grad * k * sg * (1.0 - sg), None


SURROGATE_FNS = {
    "fast_sigmoid": FastSigmoid.apply,
    "atan":         ATan.apply,
    "sigmoid":      Sigmoid.apply,
}


class LIFNeuron(nn.Module):
    def __init__(self, beta: float = 0.9, threshold: float = 1.0,
                 spike_grad_name: str = "fast_sigmoid"):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.spike_fn = SURROGATE_FNS[spike_grad_name]
        self.v = None

    def reset_state(self):
        self.v = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.v = self.beta * self.v + x
        spike = self.spike_fn(self.v, self.threshold)
        self.v = self.v * (1.0 - spike.detach())  # hard reset
        return spike