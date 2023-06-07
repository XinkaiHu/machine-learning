from torch import optim
import torch


class MyMomentum(optim.Optimizer):
    def __init__(self, params, lr, momentum, default={}) -> None:
        super().__init__(params, default)
        self.lr = lr
        self.momentum = momentum
        self.v = []
        for param_group in self.param_groups:
            params = param_group["params"]
            self.v.append([torch.zeros_like(param.data) for param in params])

    def step(self):
        for i, param_group in enumerate(self.param_groups):
            params = param_group["params"]
            v = self.v[i]
            for j, param in enumerate(params):
                v[j] = self.momentum * v[j] - self.lr * param.grad
                param.data = param.data + v[j]
