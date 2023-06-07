from torch import optim


class NoOptimizer(optim.Optimizer):
    def __init__(self, params, default={}) -> None:
        super().__init__(params, default)
        self.param_groups = params

    def step(self):
        return
