# D:\Mahdi Alizade\Projects\slowtorch\slowtorch\nn.py

import math
import random
from slowtorch.tensor import Tensor


class Parameter(Tensor):
    def __init__(self, data):
        # Parameters always track gradients
        super().__init__(data, requires_grad=True)


class Module:
    def __init__(self):
        self._submodules = []
        self._parameters = []

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            if not hasattr(self, "_parameters"):
                super().__setattr__("_parameters", [])
            self._parameters.append(value)
        elif isinstance(value, Module):
            if not hasattr(self, "_submodules"):
                super().__setattr__("_submodules", [])
            self._submodules.append(value)

        super().__setattr__(name, value)

    def parameters(self):
        params = []
        # Collect parameters belonging directly to this module
        for p in self._parameters:
            params.append(p)

        # Collect parameters recursively from child modules
        for sub in self._submodules:
            for sub_p in sub.parameters():
                params.append(sub_p)

        return params

    def zero_grad(self):
        for p in self.parameters():
            if p.grad is not None:
                if p.shape == ():
                    p.grad = 0.0
                elif len(p.shape) == 1:
                    for i in range(len(p.grad)):
                        p.grad[i] = 0.0
                elif len(p.shape) == 2:
                    for r in range(len(p.grad)):
                        for c in range(len(p.grad[0])):
                            p.grad[r][c] = 0.0

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward pass must be implemented by subclasses")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Kaiming / PyTorch default uniform initialization bound: 1 / sqrt(in_features)
        bound = 1.0 / math.sqrt(in_features)

        # Initialize weights: shape (in_features, out_features)
        weight_data = []
        for _ in range(in_features):
            row = []
            for _ in range(out_features):
                val = random.uniform(-bound, bound)
                row.append(val)
            weight_data.append(row)

        self.weight = Parameter(weight_data)

        # Initialize bias: shape (out_features,) with zeros
        if self.use_bias:
            bias_data = []
            for _ in range(out_features):
                bias_data.append(0.0)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None

    def forward(self, x):
        # Output = x @ weight + bias
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out