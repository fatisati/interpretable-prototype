# larc.py
import torch
from torch.optim import Optimizer

class LARC(Optimizer):
    def __init__(self, optimizer, trust_coefficient=0.02, clip=True, eps=1e-8):
        self.optimizer = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps
        self.clip = clip
        self.param_groups = self.optimizer.param_groups

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)
                if p_norm > 0 and g_norm > 0:
                    local_lr = self.trust_coefficient * p_norm / (g_norm + self.eps)
                    if self.clip:
                        local_lr = min(local_lr / group['lr'], 1.0)
                    p.grad.data.mul_(local_lr)

        return self.optimizer.step()

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
