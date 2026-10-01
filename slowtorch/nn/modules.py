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
        self._named_parameters = {}
        self._named_submodules = {}
        self.training = True

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

    def train(self, mode=True):
        self.training = mode
        for sub in self._named_submodules.values():
            sub.train(mode)
        return self

    def eval(self):
        return self.train(False)

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
                p.grad = _zeros_like_shape(p.shape)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward pass must be implemented by subclasses")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self._layers = []
        for idx, mod in enumerate(modules):
            if not isinstance(mod, Module):
                raise TypeError("Sequential arguments must be instances of Module, got " + str(type(mod)))
            setattr(self, str(idx), mod)
            self._layers.append(mod)

    def __getitem__(self, idx):
        return self._layers[idx]

    def __len__(self):
        return len(self._layers)

    def forward(self, x):
        out = x
        for layer in self._layers:
            out = layer(out)
        return out


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


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)

        self.use_bias = bias

        kh, kw = self.kernel_size
        fan_in = in_channels * kh * kw
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 1.0

        # Weight shape: (out_channels, in_channels, kh, kw)
        weight_data = []
        for _ in range(out_channels):
            c_in_cube = []
            for _ in range(in_channels):
                plane = []
                for _ in range(kh):
                    row = []
                    for _ in range(kw):
                        val = random.uniform(-bound, bound)
                        row.append(val)
                    plane.append(row)
                c_in_cube.append(plane)
            weight_data.append(c_in_cube)

        self.weight = Parameter(weight_data)

        if self.use_bias:
            bias_data = [0.0] * out_channels
            self.bias = Parameter(bias_data)
        else:
            self.bias = None

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError("Conv2d expects 4D input of shape (batch, in_channels, height, width), got shape " + str(x.shape))

        batch_size, in_c, in_h, in_w = x.shape
        if in_c != self.in_channels:
            raise ValueError("Input channels (" + str(in_c) + ") does not match layer in_channels (" + str(self.in_channels) + ")")

        kh, kw = self.kernel_size
        sh, sw = self.stride
        pad_h, pad_w = self.padding

        out_h = (in_h + 2 * pad_h - kh) // sh + 1
        out_w = (in_w + 2 * pad_w - kw) // sw + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError("Calculated Conv2d output dimension is non-positive: (" + str(out_h) + ", " + str(out_w) + ")")

        # Padded input preparation
        padded_h = in_h + 2 * pad_h
        padded_w = in_w + 2 * pad_w

        padded_input = []
        for n in range(batch_size):
            n_cube = []
            for c in range(in_c):
                c_plane = []
                for h in range(padded_h):
                    row = []
                    for w in range(padded_w):
                        orig_h = h - pad_h
                        orig_w = w - pad_w
                        if 0 <= orig_h < in_h and 0 <= orig_w < in_w:
                            row.append(x.data[n][c][orig_h][orig_w])
                        else:
                            row.append(0.0)
                    c_plane.append(row)
                n_cube.append(c_plane)
            padded_input.append(n_cube)

        # Forward convolution computation
        out_data = []
        for n in range(batch_size):
            n_out = []
            for cout in range(self.out_channels):
                cout_plane = []
                bias_val = self.bias.data[cout] if self.bias is not None else 0.0
                for oh in range(out_h):
                    row = []
                    for ow in range(out_w):
                        h_start = oh * sh
                        w_start = ow * sw
                        acc = bias_val
                        for cin in range(in_c):
                            for rk in range(kh):
                                for ck in range(kw):
                                    in_val = padded_input[n][cin][h_start + rk][w_start + ck]
                                    w_val = self.weight.data[cout][cin][rk][ck]
                                    acc = acc + in_val * w_val
                        row.append(acc)
                    cout_plane.append(row)
                n_out.append(cout_plane)
            out_data.append(n_out)

        parents = [x, self.weight]
        req_grad = x.requires_grad or self.weight.requires_grad
        if self.bias is not None:
            parents.append(self.bias)
            req_grad = req_grad or self.bias.requires_grad

        out = Tensor(out_data, requires_grad=req_grad, _parents=tuple(parents), _op="conv2d")

        def _backward():
            # 1. Gradient with respect to Bias
            if self.bias is not None and self.bias.requires_grad:
                if self.bias.grad is None:
                    self.bias.grad = [0.0] * self.out_channels
                for cout in range(self.out_channels):
                    bias_acc = 0.0
                    for n in range(batch_size):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                bias_acc = bias_acc + out.grad[n][cout][oh][ow]
                    self.bias.grad[cout] = self.bias.grad[cout] + bias_acc

            # 2. Gradient with respect to Weights
            if self.weight.requires_grad:
                if self.weight.grad is None:
                    self.weight.grad = _zeros_like_shape(self.weight.shape)
                for cout in range(self.out_channels):
                    for cin in range(in_c):
                        for rk in range(kh):
                            for ck in range(kw):
                                dw_acc = 0.0
                                for n in range(batch_size):
                                    for oh in range(out_h):
                                        h_in = oh * sh + rk
                                        for ow in range(out_w):
                                            w_in = ow * sw + ck
                                            og = out.grad[n][cout][oh][ow]
                                            dw_acc = dw_acc + og * padded_input[n][cin][h_in][w_in]
                                self.weight.grad[cout][cin][rk][ck] = self.weight.grad[cout][cin][rk][ck] + dw_acc

            # 3. Gradient with respect to Input x
            if x.requires_grad:
                if x.grad is None:
                    x.grad = _zeros_like_shape(x.shape)

                # Accumulate back into padded coordinates, then strip padding
                for n in range(batch_size):
                    for cout in range(self.out_channels):
                        for oh in range(out_h):
                            h_start = oh * sh
                            for ow in range(out_w):
                                w_start = ow * sw
                                og = out.grad[n][cout][oh][ow]
                                for cin in range(in_c):
                                    for rk in range(kh):
                                        h_actual = h_start + rk - pad_h
                                        if 0 <= h_actual < in_h:
                                            for ck in range(kw):
                                                w_actual = w_start + ck - pad_w
                                                if 0 <= w_actual < in_w:
                                                    w_val = self.weight.data[cout][cin][rk][ck]
                                                    x.grad[n][cin][h_actual][w_actual] = x.grad[n][cin][h_actual][w_actual] + og * w_val

        out._backward = _backward
        return out


class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = elementwise_affine

        if len(self.normalized_shape) != 1:
            raise NotImplementedError("LayerNorm currently only supports 1D normalized_shape, got " + str(self.normalized_shape))

        num_features = self.normalized_shape[0]

        if self.elementwise_affine:
            gamma_data = [1.0] * num_features
            beta_data = [0.0] * num_features
            self.weight = Parameter(gamma_data)
            self.bias = Parameter(beta_data)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x):
        if len(x.shape) != 2:
            raise NotImplementedError("LayerNorm currently only supports 2D inputs (batch_size, features)")

        rows = x.shape[0]
        cols = x.shape[1]
        dim = float(cols)

        if cols != self.normalized_shape[0]:
            raise ValueError("Input feature size (" + str(cols) + ") doesn't match LayerNorm shape (" + str(self.normalized_shape[0]) + ")")

        normalized_grid = []
        x_hat_grid = []
        inv_std_list = []

        for r in range(rows):
            mean_val = 0.0
            for c in range(cols):
                mean_val = mean_val + x.data[r][c]
            mean_val = mean_val / dim

            var_val = 0.0
            for c in range(cols):
                diff = x.data[r][c] - mean_val
                var_val = var_val + diff * diff
            var_val = var_val / dim

            inv_std = 1.0 / math.sqrt(var_val + self.eps)
            inv_std_list.append(inv_std)

            out_row = []
            x_hat_row = []
            for c in range(cols):
                x_hat = (x.data[r][c] - mean_val) * inv_std
                x_hat_row.append(x_hat)

                val = x_hat
                if self.elementwise_affine:
                    val = val * self.weight.data[c] + self.bias.data[c]
                out_row.append(val)

            x_hat_grid.append(x_hat_row)
            normalized_grid.append(out_row)

        parents = [x]
        req_grad = x.requires_grad
        if self.elementwise_affine:
            parents.append(self.weight)
            parents.append(self.bias)
            req_grad = req_grad or self.weight.requires_grad or self.bias.requires_grad

        out = Tensor(normalized_grid, requires_grad=req_grad, _parents=tuple(parents), _op="layernorm")

        def _backward():
            if self.elementwise_affine:
                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = [0.0] * cols
                    for c in range(cols):
                        g_sum = 0.0
                        for r in range(rows):
                            g_sum = g_sum + out.grad[r][c] * x_hat_grid[r][c]
                        self.weight.grad[c] = self.weight.grad[c] + g_sum

                if self.bias.requires_grad:
                    if self.bias.grad is None:
                        self.bias.grad = [0.0] * cols
                    for c in range(cols):
                        b_sum = 0.0
                        for r in range(rows):
                            b_sum = b_sum + out.grad[r][c]
                        self.bias.grad[c] = self.bias.grad[c] + b_sum

            if x.requires_grad:
                if x.grad is None:
                    x.grad = _zeros_like_shape(x.shape)

                for r in range(rows):
                    inv_std = inv_std_list[r]

                    dl_dxhat = []
                    for c in range(cols):
                        if self.elementwise_affine:
                            dl_dxhat.append(out.grad[r][c] * self.weight.data[c])
                        else:
                            dl_dxhat.append(out.grad[r][c])

                    sum_dl = 0.0
                    sum_dl_xhat = 0.0
                    for c in range(cols):
                        sum_dl = sum_dl + dl_dxhat[c]
                        sum_dl_xhat = sum_dl_xhat + dl_dxhat[c] * x_hat_grid[r][c]

                    for c in range(cols):
                        grad_term = dim * dl_dxhat[c] - sum_dl - x_hat_grid[r][c] * sum_dl_xhat
                        dx = (inv_std / dim) * grad_term
                        x.grad[r][c] = x.grad[r][c] + dx

        out._backward = _backward
        return out


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError("Dropout probability p must be in the range [0.0, 1.0), got " + str(p))
        self.p = float(p)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x

        scale = 1.0 / (1.0 - self.p)

        if len(x.shape) == 1:
            mask = []
            out_data = []
            for i in range(len(x.data)):
                keep = 1.0 if random.random() >= self.p else 0.0
                mask.append(keep * scale)
                out_data.append(x.data[i] * keep * scale)

            out = Tensor(out_data, requires_grad=x.requires_grad, _parents=(x,), _op="dropout")

            def _backward():
                if x.requires_grad:
                    if x.grad is None:
                        x.grad = [0.0] * len(x.data)
                    for i in range(len(x.data)):
                        x.grad[i] = x.grad[i] + out.grad[i] * mask[i]

            out._backward = _backward
            return out

        elif len(x.shape) == 2:
            rows = x.shape[0]
            cols = x.shape[1]
            mask = []
            out_data = []

            for r in range(rows):
                mask_row = []
                data_row = []
                for c in range(cols):
                    keep = 1.0 if random.random() >= self.p else 0.0
                    mask_row.append(keep * scale)
                    data_row.append(x.data[r][c] * keep * scale)
                mask.append(mask_row)
                out_data.append(data_row)

            out = Tensor(out_data, requires_grad=x.requires_grad, _parents=(x,), _op="dropout")

            def _backward():
                if x.requires_grad:
                    if x.grad is None:
                        x.grad = _zeros_like_shape(x.shape)
                    for r in range(rows):
                        for c in range(cols):
                            x.grad[r][c] = x.grad[r][c] + out.grad[r][c] * mask[r][c]

            out._backward = _backward
            return out
        else:
            raise NotImplementedError("Dropout currently only supports 1D and 2D tensors, got shape " + str(x.shape))


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