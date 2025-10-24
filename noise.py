import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        tensor = tensor + noise
        return torch.clamp(tensor, 0., 1.)  # 保证仍然在[0,1]区间