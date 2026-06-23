import torch

class STEBernoulli(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p):
        return torch.bernoulli(p)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

ste_bernoulli = STEBernoulli.apply


def encode_const(x: torch.Tensor, T: int):
    x = x.unsqueeze(0)
    x = x.expand(T, -1, -1, -1, -1)
    return x

def encode_rate(x: torch.Tensor, T: int):
    rate = [ste_bernoulli(x) for _ in range(T)]
    return torch.stack(rate, dim=0)

def encode_hypergeometric(x: torch.Tensor, T: int):
    hyper = []
    budget = (x * T).round()
    for t in range(T):
        prob = (budget / (T - t)).clamp(0.0, 1.0)
        fire = ste_bernoulli(prob)
        hyper.append(fire)
        budget = budget - fire.detach()
    return torch.stack(hyper, dim=0)

def get_encoder(name: str, T: int):
    if name == "const":
        return lambda x: encode_const(x, T)
    elif name == "rate":
        return lambda x: encode_rate(x, T)
    elif name == "hypergeometric":
        return lambda x: encode_hypergeometric(x, T)
    else:
        raise ValueError(f"Unknown encoding: {name}")