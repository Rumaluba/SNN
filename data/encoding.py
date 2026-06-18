import torch

def encode_const (x: torch.Tensor, T: int):
    x = x.unsqueeze(0)
    x = x.expand(T, -1, -1, -1, -1)
    return x

def encode_rate (x: torch.Tensor, T: int):
    x_c = x.clamp(0.0, 1.0)
    rate = []
    for t in range(T):
        rate.append(torch.bernoulli(x_c))
    rate = torch.stack (rate, dim=0)
    return rate

def encode_hypergeometric(x: torch.Tensor, T: int):
    x_c = x.clamp(0.0, 1.0)
    hyper = []
    budget = (x_c * T).round()
    for t in range(T):
        prob = (budget / (T - t)).clamp(0.0, 1.0)
        fire = torch.bernoulli(prob)
        hyper.append(fire)
        budget -= fire
    hyper = torch.stack (hyper, dim=0)
    return hyper

def get_encoder(name: str, T: int):
    if name == "const":
        return lambda x: encode_const(x, T)
    elif name == "rate":
        return lambda x: encode_rate(x, T)
    elif name == "hypergeometric":
        return lambda x: encode_hypergeometric(x, T)
    else:
        raise ValueError(f"Unknown encoding: {name}")