import math
import random
import json
from slowtorch.tensor import Tensor, _zeros_like_shape


def _deep_copy_nested_list(data):
    if isinstance(data, (int, float)):
        return float(data)
    if isinstance(data, list):
        res = []
        for item in data:
            res.append(_deep_copy_nested_list(item))
        return res
    return data


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    def __init__(self):
        # Using dicts with insertion order to track named parameters and submodules
        self._named_parameters = {}
        self._named_submodules = {}

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            if not hasattr(self, "_named_parameters"):
                super().__setattr__("_named_parameters", {})
            self._named_parameters[name] = value
        elif isinstance(value, Module):
            if not hasattr(self, "_named_submodules"):
                super().__setattr__("_named_submodules", {})
            self._named_submodules[name] = value

        super().__setattr__(name, value)

    def parameters(self):
        params = []
        for p in self._named_parameters.values():
            params.append(p)

        for sub in self._named_submodules.values():
            for sub_p in sub.parameters():
                params.append(sub_p)

        return params

    def named_parameters(self, prefix=""):
        items = []
        for name, param in self._named_parameters.items():
            full_name = (prefix + "." + name) if prefix else name
            items.append((full_name, param))

        for sub_name, sub in self._named_submodules.items():
            sub_prefix = (prefix + "." + sub_name) if prefix else sub_name
            for full_name, param in sub.named_parameters(prefix=sub_prefix):
                items.append((full_name, param))

        return items

    def state_dict(self):
        state = {}
        for name, param in self.named_parameters():
            state[name] = _deep_copy_nested_list(param.data)
        return state

    def load_state_dict(self, state_dict, strict=True):
        current_params = dict(self.named_parameters())
        state_keys = set(state_dict.keys())
        model_keys = set(current_params.keys())

        if strict:
            missing_keys = model_keys - state_keys
            unexpected_keys = state_keys - model_keys
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                err_msg = "Error loading state_dict."
                if len(missing_keys) > 0:
                    err_msg = err_msg + " Missing keys: " + str(list(missing_keys)) + "."
                if len(unexpected_keys) > 0:
                    err_msg = err_msg + " Unexpected keys: " + str(list(unexpected_keys)) + "."
                raise KeyError(err_msg)

        for name, target_param in current_params.items():
            if name in state_dict:
                incoming_data = state_dict[name]
                # Replace data with deep copied incoming values
                target_param.data = _deep_copy_nested_list(incoming_data)

    def save(self, filepath):
        state = self.state_dict()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load(self, filepath, strict=True):
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.load_state_dict(state, strict=strict)

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

        bound = 1.0 / math.sqrt(in_features)

        weight_data = []
        for _ in range(in_features):
            row = []
            for _ in range(out_features):
                val = random.uniform(-bound, bound)
                row.append(val)
            weight_data.append(row)

        self.weight = Parameter(weight_data)

        if self.use_bias:
            bias_data = []
            for _ in range(out_features):
                bias_data.append(0.0)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.relu()


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sigmoid()


class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        diff = pred - target
        squared = diff ** 2
        return squared.mean()


class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if len(x.shape) != 2:
            raise NotImplementedError("Softmax currently only supports 2D tensors (batch_size, num_classes)")

        rows = x.shape[0]
        cols = x.shape[1]
        probs = []

        for r in range(rows):
            max_val = x.data[r][0]
            for c in range(1, cols):
                if x.data[r][c] > max_val:
                    max_val = x.data[r][c]

            exp_vals = []
            exp_sum = 0.0
            for c in range(cols):
                e = math.exp(x.data[r][c] - max_val)
                exp_vals.append(e)
                exp_sum = exp_sum + e

            row_prob = []
            for c in range(cols):
                row_prob.append(exp_vals[c] / exp_sum)
            probs.append(row_prob)

        out = Tensor(probs, requires_grad=x.requires_grad, _parents=(x,), _op="softmax")

        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = _zeros_like_shape(x.shape)
                for r in range(rows):
                    dot_grad_p = 0.0
                    for c in range(cols):
                        dot_grad_p = dot_grad_p + out.grad[r][c] * probs[r][c]
                    for c in range(cols):
                        local_grad = probs[r][c] * (out.grad[r][c] - dot_grad_p)
                        x.grad[r][c] = x.grad[r][c] + local_grad

        out._backward = _backward
        return out


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        if len(logits.shape) != 2:
            raise ValueError("Logits must be 2D tensor (batch_size, num_classes)")

        rows = logits.shape[0]
        cols = logits.shape[1]

        if isinstance(targets, Tensor):
            if len(targets.shape) == 2:
                target_indices = [int(targets.data[i][0]) for i in range(rows)]
            else:
                target_indices = [int(targets.data[i]) for i in range(rows)]
        elif isinstance(targets, list):
            target_indices = [int(idx) for idx in targets]
        else:
            raise TypeError("Unsupported targets format for CrossEntropyLoss")

        total_loss = 0.0
        probabilities = []

        for r in range(rows):
            max_val = logits.data[r][0]
            for c in range(1, cols):
                if logits.data[r][c] > max_val:
                    max_val = logits.data[r][c]

            exp_vals = []
            sum_exp = 0.0
            for c in range(cols):
                shift = logits.data[r][c] - max_val
                e = math.exp(shift)
                exp_vals.append(e)
                sum_exp = sum_exp + e

            log_sum_exp = max_val + math.log(sum_exp)
            correct_class = target_indices[r]
            loss_item = log_sum_exp - logits.data[r][correct_class]
            total_loss = total_loss + loss_item

            row_prob = []
            for c in range(cols):
                row_prob.append(exp_vals[c] / sum_exp)
            probabilities.append(row_prob)

        mean_loss = total_loss / float(rows)
        out = Tensor(mean_loss, requires_grad=logits.requires_grad, _parents=(logits,), _op="cross_entropy")

        def _backward():
            if logits.requires_grad:
                if logits.grad is None:
                    logits.grad = _zeros_like_shape(logits.shape)
                for r in range(rows):
                    correct_class = target_indices[r]
                    for c in range(cols):
                        p = probabilities[r][c]
                        if c == correct_class:
                            grad_val = (p - 1.0) / float(rows)
                        else:
                            grad_val = p / float(rows)
                        logits.grad[r][c] = logits.grad[r][c] + grad_val * out.grad

        out._backward = _backward
        return out