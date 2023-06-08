from torch import optim
import torch


class MyAdam(optim.Optimizer):
    def __init__(self, params, lr, beta1, beta2, epsilon, default={}) -> None:
        super().__init__(params, default)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.v = []
        self.m = []
        for param_group in self.param_groups:
            params = param_group["params"]
            self.v.append([torch.zeros_like(param.data) for param in params])
            self.m.append([torch.zeros_like(param.data) for param in params])

    def step(self):
        for i, param_group in enumerate(self.param_groups):
            params = param_group["params"]
            m = self.m[i]
            v = self.v[i]
            for j, param in enumerate(params):
                m[j] = self.beta1 * m[j] + (1 - self.beta1) * param.grad
                v[j] = self.beta2 * v[j] + (1 - self.beta2) * torch.square(param.grad)
                param.data -= self.lr * m[j].div(self.epsilon + torch.sqrt(v[j]))
