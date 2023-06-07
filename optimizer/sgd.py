from torch import optim


class MySGD(optim.Optimizer):
    def __init__(self, params, lr, default={}) -> None:
        super().__init__(params, default)
        self.lr = lr

    def step(self):
        for param_group in self.param_groups:
            params = param_group["params"]
            for param in params:
                param.data -= self.lr * param.grad
