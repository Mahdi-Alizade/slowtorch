import math
import random
import json
from slowtorch.tensor import Tensor, _zeros_like_shape, _flatten_list


def _deep_copy_nested_list(data):
    if isinstance(data, (int, float)):
        return float(data)
    if isinstance(data, list):
        res = []
        for item in data:
            res.append(_deep_copy_nested_list(item))
        return res
    return data


def _unflatten_to_original_shape(flat_list, shape):
    if len(shape) == 0:
        return flat_list[0]
    if len(shape) == 1:
        return list(flat_list)
    if len(shape) == 2:
        rows, cols = shape
        out = []
        idx = 0
        for _ in range(rows):
            row = []
            for _ in range(cols):
                row.append(flat_list[idx])
                idx = idx + 1
            out.append(row)
        return out
    if len(shape) == 3:
        d0, d1, d2 = shape
        out3d = []
        idx = 0
        for _ in range(d0):
            plane = []
            for _ in range(d1):
                row = []
                for _ in range(d2):
                    row.append(flat_list[idx])
                    idx = idx + 1
                plane.append(row)
            out3d.append(plane)
        return out3d
    if len(shape) == 4:
        d0, d1, d2, d3 = shape
        out4d = []
        idx = 0
        for _ in range(d0):
            cube = []
            for _ in range(d1):
                plane = []
                for _ in range(d2):
                    row = []
                    for _ in range(d3):
                        row.append(flat_list[idx])
                        idx = idx + 1
                    plane.append(row)
                cube.append(plane)
            out4d.append(cube)
        return out4d
    raise NotImplementedError("Unflattening not implemented for shapes above 4D: " + str(shape))


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


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
        if len(x.shape) <= 1:
            return x

        batch_size = x.shape[0]
        flattened_rows = []

        for n in range(batch_size):
            sample = x.data[n]
            flat_sample = _flatten_list(sample)
            flattened_rows.append(flat_sample)

        out = Tensor(
            flattened_rows,
            requires_grad=x.requires_grad,
            _parents=(x,),
            _op="flatten"
        )

        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = _zeros_like_shape(x.shape)

                flat_out_grad = _flatten_list(out.grad)
                reconstructed_grad = _unflatten_to_original_shape(flat_out_grad, x.shape)

                def _accumulate(dest, src):
                    if isinstance(dest, list):
                        for i in range(len(dest)):
                            if isinstance(dest[i], list):
                                _accumulate(dest[i], src[i])
                            else:
                                dest[i] = dest[i] + src[i]

                _accumulate(x.grad, reconstructed_grad)

        out._backward = _backward
        return out


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.padding_idx = padding_idx if padding_idx is None else int(padding_idx)

        weight_data = []
        for row_idx in range(self.num_embeddings):
            if self.padding_idx is not None and row_idx == self.padding_idx:
                row = [0.0] * self.embedding_dim
            else:
                row = []
                for _ in range(self.embedding_dim):
                    val = random.gauss(0.0, 1.0)
                    row.append(val)
            weight_data.append(row)

        self.weight = Parameter(weight_data)

    def forward(self, indices):
        if isinstance(indices, Tensor):
            idx_data = indices.data
            idx_shape = indices.shape
        elif isinstance(indices, list):
            idx_data = indices
            temp = Tensor(indices)
            idx_shape = temp.shape
        else:
            raise TypeError("Embedding indices must be a Tensor or list of integers, got " + str(type(indices)))

        dim = self.embedding_dim

        if len(idx_shape) == 1:
            seq_len = idx_shape[0]
            lookup_out = []
            for i in range(seq_len):
                w_idx = int(idx_data[i])
                if not (0 <= w_idx < self.num_embeddings):
                    raise IndexError("Index " + str(w_idx) + " out of range for Embedding of size " + str(self.num_embeddings))
                row_copy = []
                for d in range(dim):
                    row_copy.append(self.weight.data[w_idx][d])
                lookup_out.append(row_copy)

            out = Tensor(lookup_out, requires_grad=self.weight.requires_grad, _parents=(self.weight,), _op="embedding")

            def _backward():
                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = _zeros_like_shape(self.weight.shape)
                    for i in range(seq_len):
                        w_idx = int(idx_data[i])
                        if self.padding_idx is not None and w_idx == self.padding_idx:
                            continue
                        for d in range(dim):
                            self.weight.grad[w_idx][d] = self.weight.grad[w_idx][d] + out.grad[i][d]

            out._backward = _backward
            return out

        elif len(idx_shape) == 2:
            batch_size = idx_shape[0]
            seq_len = idx_shape[1]
            lookup_out = []

            for b in range(batch_size):
                b_plane = []
                for l in range(seq_len):
                    w_idx = int(idx_data[b][l])
                    if not (0 <= w_idx < self.num_embeddings):
                        raise IndexError("Index " + str(w_idx) + " out of range for Embedding of size " + str(self.num_embeddings))
                    row_copy = []
                    for d in range(dim):
                        row_copy.append(self.weight.data[w_idx][d])
                    b_plane.append(row_copy)
                lookup_out.append(b_plane)

            out = Tensor(lookup_out, requires_grad=self.weight.requires_grad, _parents=(self.weight,), _op="embedding")

            def _backward():
                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = _zeros_like_shape(self.weight.shape)
                    for b in range(batch_size):
                        for l in range(seq_len):
                            w_idx = int(idx_data[b][l])
                            if self.padding_idx is not None and w_idx == self.padding_idx:
                                continue
                            for d in range(dim):
                                self.weight.grad[w_idx][d] = self.weight.grad[w_idx][d] + out.grad[b][l][d]

            out._backward = _backward
            return out

        else:
            raise NotImplementedError("Embedding currently supports 1D and 2D index tensors, got shape " + str(idx_shape))


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

            if x.requires_grad:
                if x.grad is None:
                    x.grad = _zeros_like_shape(x.shape)

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


class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)

        if stride is None:
            self.stride = self.kernel_size
        elif isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = tuple(stride)

        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError("MaxPool2d expects 4D input of shape (batch, channels, height, width), got shape " + str(x.shape))

        batch_size, channels, in_h, in_w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        pad_h, pad_w = self.padding

        out_h = (in_h + 2 * pad_h - kh) // sh + 1
        out_w = (in_w + 2 * pad_w - kw) // sw + 1

        if out_h <= 0 or out_w <= 0:
            raise ValueError("Calculated MaxPool2d output dimension is non-positive: (" + str(out_h) + ", " + str(out_w) + ")")

        argmax_mask = []
        out_data = []

        for n in range(batch_size):
            n_out = []
            n_mask = []
            for c in range(channels):
                c_out = []
                c_mask = []
                for oh in range(out_h):
                    row_out = []
                    row_mask = []
                    for ow in range(out_w):
                        h_start = oh * sh - pad_h
                        w_start = ow * sw - pad_w

                        max_val = -float("inf")
                        max_idx = (None, None)

                        for rk in range(kh):
                            curr_h = h_start + rk
                            for ck in range(kw):
                                curr_w = w_start + ck
                                if 0 <= curr_h < in_h and 0 <= curr_w < in_w:
                                    val = x.data[n][c][curr_h][curr_w]
                                else:
                                    val = -float("inf")

                                if val > max_val:
                                    max_val = val
                                    max_idx = (curr_h, curr_w)

                        row_out.append(max_val)
                        row_mask.append(max_idx)

                    c_out.append(row_out)
                    c_mask.append(row_mask)
                n_out.append(c_out)
                n_mask.append(c_mask)
            out_data.append(n_out)
            argmax_mask.append(n_mask)

        out = Tensor(out_data, requires_grad=x.requires_grad, _parents=(x,), _op="maxpool2d")

        def _backward():
            if x.requires_grad:
                if x.grad is None:
                    x.grad = _zeros_like_shape(x.shape)

                for n in range(batch_size):
                    for c in range(channels):
                        for oh in range(out_h):
                            for ow in range(out_w):
                                best_h, best_w = argmax_mask[n][c][oh][ow]
                                if best_h is not None and best_w is not None:
                                    og = out.grad[n][c][oh][ow]
                                    x.grad[n][c][best_h][best_w] = x.grad[n][c][best_h][best_w] + og

        out._backward = _backward
        return out


class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)

        if self.affine:
            self.weight = Parameter([1.0] * self.num_features)
            self.bias = Parameter([0.0] * self.num_features)
        else:
            self.weight = None
            self.bias = None

        if self.track_running_stats:
            self.running_mean = [0.0] * self.num_features
            self.running_var = [1.0] * self.num_features
        else:
            self.running_mean = None
            self.running_var = None

    def state_dict(self):
        state = super().state_dict()
        if self.track_running_stats:
            state["running_mean"] = _deep_copy_nested_list(self.running_mean)
            state["running_var"] = _deep_copy_nested_list(self.running_var)
        return state

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        if self.track_running_stats:
            if "running_mean" in state_dict:
                self.running_mean = _deep_copy_nested_list(state_dict["running_mean"])
            if "running_var" in state_dict:
                self.running_var = _deep_copy_nested_list(state_dict["running_var"])

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError("BatchNorm2d expects 4D input of shape (batch, channels, height, width), got shape " + str(x.shape))

        batch_size, channels, height, width = x.shape
        if channels != self.num_features:
            raise ValueError("Input channels (" + str(channels) + ") does not match BatchNorm2d num_features (" + str(self.num_features) + ")")

        # Total elements per channel across batch and spatial dims
        m = float(batch_size * height * width)

        out_data = []
        x_hat_data = []
        inv_std_list = []

        if self.training or not self.track_running_stats:
            # 1. Compute batch mean and batch variance per channel
            means = []
            vars_ = []

            for c in range(channels):
                sum_val = 0.0
                for n in range(batch_size):
                    for h in range(height):
                        for w in range(width):
                            sum_val = sum_val + x.data[n][c][h][w]
                mean_c = sum_val / m
                means.append(mean_c)

                sq_sum = 0.0
                for n in range(batch_size):
                    for h in range(height):
                        for w in range(width):
                            diff = x.data[n][c][h][w] - mean_c
                            sq_sum = sq_sum + diff * diff
                var_c = sq_sum / m
                vars_.append(var_c)

                # Update running statistics with exponential moving average
                if self.training and self.track_running_stats:
                    # Unbiased sample variance for running_var (Bessel correction)
                    unbiased_var = (m / (m - 1.0)) * var_c if m > 1.0 else var_c
                    self.running_mean[c] = (1.0 - self.momentum) * self.running_mean[c] + self.momentum * mean_c
                    self.running_var[c] = (1.0 - self.momentum) * self.running_var[c] + self.momentum * unbiased_var

            # 2. Compute normalized values and output
            for n in range(batch_size):
                n_out = []
                n_xhat = []
                for c in range(channels):
                    inv_std = 1.0 / math.sqrt(vars_[c] + self.eps)
                    if n == 0:
                        inv_std_list.append(inv_std)
                    gamma = self.weight.data[c] if self.affine else 1.0
                    beta = self.bias.data[c] if self.affine else 0.0

                    c_out = []
                    c_xhat = []
                    for h in range(height):
                        r_out = []
                        r_xhat = []
                        for w in range(width):
                            x_hat = (x.data[n][c][h][w] - means[c]) * inv_std
                            y_val = gamma * x_hat + beta
                            r_out.append(y_val)
                            r_xhat.append(x_hat)
                        c_out.append(r_out)
                        c_xhat.append(r_xhat)
                    n_out.append(c_out)
                    n_xhat.append(c_xhat)
                out_data.append(n_out)
                x_hat_data.append(n_xhat)

        else:
            # Inference mode with running statistics (Eval mode)
            for n in range(batch_size):
                n_out = []
                for c in range(channels):
                    mean_c = self.running_mean[c]
                    var_c = self.running_var[c]
                    inv_std = 1.0 / math.sqrt(var_c + self.eps)
                    gamma = self.weight.data[c] if self.affine else 1.0
                    beta = self.bias.data[c] if self.affine else 0.0

                    c_out = []
                    for h in range(height):
                        r_out = []
                        for w in range(width):
                            x_hat = (x.data[n][c][h][w] - mean_c) * inv_std
                            y_val = gamma * x_hat + beta
                            r_out.append(y_val)
                        c_out.append(r_out)
                    n_out.append(c_out)
                out_data.append(n_out)

        parents = [x]
        req_grad = x.requires_grad
        if self.affine:
            parents.append(self.weight)
            parents.append(self.bias)
            req_grad = req_grad or self.weight.requires_grad or self.bias.requires_grad

        out = Tensor(out_data, requires_grad=req_grad, _parents=tuple(parents), _op="batchnorm2d")

        is_train_mode = self.training or not self.track_running_stats

        def _backward():
            # 1. Gradients with respect to gamma (weight) and beta (bias)
            if self.affine:
                if self.bias.requires_grad:
                    if self.bias.grad is None:
                        self.bias.grad = [0.0] * channels
                    for c in range(channels):
                        d_beta = 0.0
                        for n in range(batch_size):
                            for h in range(height):
                                for w in range(width):
                                    d_beta = d_beta + out.grad[n][c][h][w]
                        self.bias.grad[c] = self.bias.grad[c] + d_beta

                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = [0.0] * channels
                    for c in range(channels):
                        d_gamma = 0.0
                        for n in range(batch_size):
                            for h in range(height):
                                for w in range(width):
                                    if is_train_mode:
                                        x_hat_val = x_hat_data[n][c][h][w]
                                    else:
                                        x_hat_val = (x.data[n][c][h][w] - self.running_mean[c]) / math.sqrt(self.running_var[c] + self.eps)
                                    d_gamma = d_gamma + out.grad[n][c][h][w] * x_hat_val
                        self.weight.grad[c] = self.weight.grad[c] + d_gamma

            # 2. Gradient with respect to input x
            if x.requires_grad:
                if x.grad is None:
                    x.grad = _zeros_like_shape(x.shape)

                if is_train_mode:
                    for c in range(channels):
                        gamma = self.weight.data[c] if self.affine else 1.0
                        inv_std = inv_std_list[c]

                        sum_dl_dxhat = 0.0
                        sum_dl_xhat_dot = 0.0

                        for n in range(batch_size):
                            for h in range(height):
                                for w in range(width):
                                    dl_dxhat = out.grad[n][c][h][w] * gamma
                                    x_hat_val = x_hat_data[n][c][h][w]
                                    sum_dl_dxhat = sum_dl_dxhat + dl_dxhat
                                    sum_dl_xhat_dot = sum_dl_xhat_dot + dl_dxhat * x_hat_val

                        factor = inv_std / m
                        for n in range(batch_size):
                            for h in range(height):
                                for w in range(width):
                                    dl_dxhat = out.grad[n][c][h][w] * gamma
                                    x_hat_val = x_hat_data[n][c][h][w]
                                    dx = factor * (m * dl_dxhat - sum_dl_dxhat - x_hat_val * sum_dl_xhat_dot)
                                    x.grad[n][c][h][w] = x.grad[n][c][h][w] + dx
                else:
                    # In eval mode, normalization depends solely on fixed running stats
                    for c in range(channels):
                        gamma = self.weight.data[c] if self.affine else 1.0
                        inv_std = 1.0 / math.sqrt(self.running_var[c] + self.eps)
                        for n in range(batch_size):
                            for h in range(height):
                                for w in range(width):
                                    dx = out.grad[n][c][h][w] * gamma * inv_std
                                    x.grad[n][c][h][w] = x.grad[n][c][h][w] + dx

        out._backward = _backward
        return out


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias

        bound = 1.0 / math.sqrt(hidden_size)

        w_ih_data = []
        for _ in range(input_size):
            row = []
            for _ in range(hidden_size):
                row.append(random.uniform(-bound, bound))
            w_ih_data.append(row)
        self.weight_ih = Parameter(w_ih_data)

        w_hh_data = []
        for _ in range(hidden_size):
            row = []
            for _ in range(hidden_size):
                row.append(random.uniform(-bound, bound))
            w_hh_data.append(row)
        self.weight_hh = Parameter(w_hh_data)

        if self.use_bias:
            b_ih_data = [0.0] * hidden_size
            b_hh_data = [0.0] * hidden_size
            self.bias_ih = Parameter(b_ih_data)
            self.bias_hh = Parameter(b_hh_data)
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, h=None):
        batch_size = x.shape[0]

        if h is None:
            h_zeros = [[0.0] * self.hidden_size for _ in range(batch_size)]
            h = Tensor(h_zeros, requires_grad=False)

        ih = x @ self.weight_ih
        if self.bias_ih is not None:
            ih = ih + self.bias_ih

        hh = h @ self.weight_hh
        if self.bias_hh is not None:
            hh = hh + self.bias_hh

        pre_act = ih + hh
        h_next = pre_act.tanh()
        return h_next


class RNN(Module):
    def __init__(self, input_size, hidden_size, bias=True, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first

        self.cell = RNNCell(input_size, hidden_size, bias=bias)

    def forward(self, x, h_0=None):
        if len(x.shape) != 3:
            raise ValueError("RNN expects 3D input tensor, got shape " + str(x.shape))

        if self.batch_first:
            batch_size, seq_len, in_size = x.shape
        else:
            seq_len, batch_size, in_size = x.shape

        if in_size != self.input_size:
            raise ValueError("Input feature size (" + str(in_size) + ") does not match RNN input_size (" + str(self.input_size) + ")")

        h_t = h_0
        h_seq = []

        for t in range(seq_len):
            if self.batch_first:
                step_data = [x.data[b][t] for b in range(batch_size)]
            else:
                step_data = x.data[t]

            x_t = Tensor(step_data, requires_grad=x.requires_grad, _parents=(x,), _op="slice_t")

            if x.requires_grad:
                t_idx = t
                bf = self.batch_first

                def _make_backward_slice(t_curr, b_flag, x_node, step_tensor):
                    def _backward():
                        if x_node.requires_grad and step_tensor.grad is not None:
                            if x_node.grad is None:
                                x_node.grad = _zeros_like_shape(x_node.shape)
                            for b in range(batch_size):
                                for feat in range(in_size):
                                    if b_flag:
                                        x_node.grad[b][t_curr][feat] = x_node.grad[b][t_curr][feat] + step_tensor.grad[b][feat]
                                    else:
                                        x_node.grad[t_curr][b][feat] = x_node.grad[t_curr][b][feat] + step_tensor.grad[b][feat]
                    return _backward

                x_t._backward = _make_backward_slice(t_idx, bf, x, x_t)

            h_t = self.cell(x_t, h_t)
            h_seq.append(h_t)

        if self.batch_first:
            assembled_out = []
            for b in range(batch_size):
                b_seq = []
                for t in range(seq_len):
                    b_seq.append(h_seq[t].data[b])
                assembled_out.append(b_seq)
        else:
            assembled_out = [h_step.data for h_step in h_seq]

        parents = tuple(h_seq)
        out_req_grad = any(h_step.requires_grad for h_step in h_seq)
        output = Tensor(assembled_out, requires_grad=out_req_grad, _parents=parents, _op="rnn_unroll")

        if out_req_grad:
            bf_flag = self.batch_first

            def _backward():
                for t in range(seq_len):
                    h_step = h_seq[t]
                    if h_step.requires_grad and output.grad is not None:
                        if h_step.grad is None:
                            h_step.grad = _zeros_like_shape(h_step.shape)
                        for b in range(batch_size):
                            for h_idx in range(self.hidden_size):
                                if bf_flag:
                                    h_step.grad[b][h_idx] = h_step.grad[b][h_idx] + output.grad[b][t][h_idx]
                                else:
                                    h_step.grad[b][h_idx] = h_step.grad[b][h_idx] + output.grad[t][b][h_idx]

            output._backward = _backward

        return output, h_t


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = bias

        bound = 1.0 / math.sqrt(hidden_size)

        w_ih_data = []
        for _ in range(input_size):
            row = []
            for _ in range(4 * hidden_size):
                row.append(random.uniform(-bound, bound))
            w_ih_data.append(row)
        self.weight_ih = Parameter(w_ih_data)

        w_hh_data = []
        for _ in range(hidden_size):
            row = []
            for _ in range(4 * hidden_size):
                row.append(random.uniform(-bound, bound))
            w_hh_data.append(row)
        self.weight_hh = Parameter(w_hh_data)

        if self.use_bias:
            b_ih_data = [0.0] * (4 * hidden_size)
            b_hh_data = [0.0] * (4 * hidden_size)
            self.bias_ih = Parameter(b_ih_data)
            self.bias_hh = Parameter(b_hh_data)
        else:
            self.bias_ih = None
            self.bias_hh = None

    def forward(self, x, states=None):
        batch_size = x.shape[0]
        h_dim = self.hidden_size

        if states is None:
            h_zeros = [[0.0] * h_dim for _ in range(batch_size)]
            c_zeros = [[0.0] * h_dim for _ in range(batch_size)]
            h = Tensor(h_zeros, requires_grad=False)
            c = Tensor(c_zeros, requires_grad=False)
        else:
            h, c = states

        gates = x @ self.weight_ih
        if self.bias_ih is not None:
            gates = gates + self.bias_ih

        h_proj = h @ self.weight_hh
        if self.bias_hh is not None:
            h_proj = h_proj + self.bias_hh

        gates = gates + h_proj

        i_data = []
        f_data = []
        g_data = []
        o_data = []

        for b in range(batch_size):
            row = gates.data[b]
            i_data.append(row[0 : h_dim])
            f_data.append(row[h_dim : 2 * h_dim])
            g_data.append(row[2 * h_dim : 3 * h_dim])
            o_data.append(row[3 * h_dim : 4 * h_dim])

        req_grad = gates.requires_grad
        i_gate_pre = Tensor(i_data, requires_grad=req_grad, _parents=(gates,), _op="slice_i")
        f_gate_pre = Tensor(f_data, requires_grad=req_grad, _parents=(gates,), _op="slice_f")
        g_gate_pre = Tensor(g_data, requires_grad=req_grad, _parents=(gates,), _op="slice_g")
        o_gate_pre = Tensor(o_data, requires_grad=req_grad, _parents=(gates,), _op="slice_o")

        if req_grad:
            def _backward_gates_slice():
                if gates.requires_grad:
                    if gates.grad is None:
                        gates.grad = _zeros_like_shape(gates.shape)
                    for b in range(batch_size):
                        for k in range(h_dim):
                            if i_gate_pre.grad is not None:
                                gates.grad[b][k] = gates.grad[b][k] + i_gate_pre.grad[b][k]
                            if f_gate_pre.grad is not None:
                                gates.grad[b][h_dim + k] = gates.grad[b][h_dim + k] + f_gate_pre.grad[b][k]
                            if g_gate_pre.grad is not None:
                                gates.grad[b][2 * h_dim + k] = gates.grad[b][2 * h_dim + k] + g_gate_pre.grad[b][k]
                            if o_gate_pre.grad is not None:
                                gates.grad[b][3 * h_dim + k] = gates.grad[b][3 * h_dim + k] + o_gate_pre.grad[b][k]

            i_gate_pre._backward = _backward_gates_slice
            f_gate_pre._backward = _backward_gates_slice
            g_gate_pre._backward = _backward_gates_slice
            o_gate_pre._backward = _backward_gates_slice

        i_gate = i_gate_pre.sigmoid()
        f_gate = f_gate_pre.sigmoid()
        g_gate = g_gate_pre.tanh()
        o_gate = o_gate_pre.sigmoid()

        c_next = (f_gate * c) + (i_gate * g_gate)
        h_next = o_gate * c_next.tanh()

        return h_next, c_next


class LSTM(Module):
    def __init__(self, input_size, hidden_size, bias=True, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first

        self.cell = LSTMCell(input_size, hidden_size, bias=bias)

    def forward(self, x, states_0=None):
        if len(x.shape) != 3:
            raise ValueError("LSTM expects 3D input tensor, got shape " + str(x.shape))

        if self.batch_first:
            batch_size, seq_len, in_size = x.shape
        else:
            seq_len, batch_size, in_size = x.shape

        if in_size != self.input_size:
            raise ValueError("Input feature size (" + str(in_size) + ") does not match LSTM input_size (" + str(self.input_size) + ")")

        if states_0 is None:
            h_t = None
            c_t = None
        else:
            h_t, c_t = states_0

        h_seq = []

        for t in range(seq_len):
            if self.batch_first:
                step_data = [x.data[b][t] for b in range(batch_size)]
            else:
                step_data = x.data[t]

            x_t = Tensor(step_data, requires_grad=x.requires_grad, _parents=(x,), _op="slice_t")

            if x.requires_grad:
                t_idx = t
                bf = self.batch_first

                def _make_backward_slice(t_curr, b_flag, x_node, step_tensor):
                    def _backward():
                        if x_node.requires_grad and step_tensor.grad is not None:
                            if x_node.grad is None:
                                x_node.grad = _zeros_like_shape(x_node.shape)
                            for b in range(batch_size):
                                for feat in range(in_size):
                                    if b_flag:
                                        x_node.grad[b][t_curr][feat] = x_node.grad[b][t_curr][feat] + step_tensor.grad[b][feat]
                                    else:
                                        x_node.grad[t_curr][b][feat] = x_node.grad[t_curr][b][feat] + step_tensor.grad[b][feat]
                    return _backward

                x_t._backward = _make_backward_slice(t_idx, bf, x, x_t)

            curr_states = (h_t, c_t) if (h_t is not None and c_t is not None) else None
            h_t, c_t = self.cell(x_t, curr_states)
            h_seq.append(h_t)

        if self.batch_first:
            assembled_out = []
            for b in range(batch_size):
                b_seq = []
                for t in range(seq_len):
                    b_seq.append(h_seq[t].data[b])
                assembled_out.append(b_seq)
        else:
            assembled_out = [h_step.data for h_step in h_seq]

        parents = tuple(h_seq)
        out_req_grad = any(h_step.requires_grad for h_step in h_seq)
        output = Tensor(assembled_out, requires_grad=out_req_grad, _parents=parents, _op="lstm_unroll")

        if out_req_grad:
            bf_flag = self.batch_first

            def _backward():
                for t in range(seq_len):
                    h_step = h_seq[t]
                    if h_step.requires_grad and output.grad is not None:
                        if h_step.grad is None:
                            h_step.grad = _zeros_like_shape(h_step.shape)
                        for b in range(batch_size):
                            for h_idx in range(self.hidden_size):
                                if bf_flag:
                                    h_step.grad[b][h_idx] = h_step.grad[b][h_idx] + output.grad[b][t][h_idx]
                                else:
                                    h_step.grad[b][h_idx] = h_step.grad[b][h_idx] + output.grad[t][b][h_idx]

            output._backward = _backward

        return output, (h_t, c_t)


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