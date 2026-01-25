"""
LARC (Layer-wise Adaptive Rate Clipping) optimizer wrapper.
Drop-in replacement for apex.parallel.LARC that doesn't require apex.

Based on the LARS/LARC algorithm from:
"Large Batch Training of Convolutional Networks" (https://arxiv.org/abs/1708.03888)
"""
import torch
from torch.optim.optimizer import Optimizer


class LARC(Optimizer):
    """
    Layer-wise Adaptive Rate Clipping optimizer wrapper.

    Wraps an existing optimizer and applies LARC scaling to gradients.

    Args:
        optimizer: Base optimizer (usually SGD)
        trust_coefficient: Trust ratio clipping coefficient (default: 0.001)
        clip: If True, clip the ratio; if False, scale without clipping (default: False)
        eps: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(self, optimizer, trust_coefficient=0.001, clip=False, eps=1e-8):
        self.optimizer = optimizer
        self.trust_coefficient = trust_coefficient
        self.clip = clip
        self.eps = eps

        # Expose param_groups from the wrapped optimizer
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

    def __getstate__(self):
        return {
            'optimizer': self.optimizer,
            'trust_coefficient': self.trust_coefficient,
            'clip': self.clip,
            'eps': self.eps,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def add_param_group(self, param_group):
        self.optimizer.add_param_group(param_group)

    def step(self, closure=None):
        """Performs a single optimization step with LARC scaling."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.optimizer.param_groups:
            weight_decay = group.get('weight_decay', 0)
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_norm = p.data.norm(2)
                grad_norm = p.grad.data.norm(2)

                # Compute local learning rate
                if param_norm > 0 and grad_norm > 0:
                    # LARC trust ratio
                    local_lr = self.trust_coefficient * param_norm / (
                        grad_norm + weight_decay * param_norm + self.eps
                    )

                    if self.clip:
                        # Clip the local lr to not exceed the base lr
                        local_lr = min(local_lr, lr)

                    # Scale the gradient
                    p.grad.data.mul_(local_lr / lr)

        # Perform the actual optimization step
        self.optimizer.step()

        return loss
