import math
from functools import wraps

_grad_enabled = True


def is_grad_enabled():
    global _grad_enabled
    return _grad_enabled


def set_grad_enabled(mode):
    global _grad_enabled
    _grad_enabled = bool(mode)


class no_grad:
    def __init__(self):
        self.prev = True

    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _grad_enabled
        _grad_enabled = self.prev

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper


def _zeros_like_shape(shape):
    if len(shape) == 0:
        return 0.0
    dim = shape[0]
    sub_shape = shape[1:]
    res = []
    for _ in range(dim):
        res.append(_zeros_like_shape(sub_shape))
    return res


def _flatten_list(nested):
    if not isinstance(nested, list):
        return [nested]
    flat = []
    for item in nested:
        if isinstance(item, list):
            flat.extend(_flatten_list(item))
        else:
            flat.append(item)
    return flat


def _unflatten_to_shape(flat_list, target_shape):
    if len(target_shape) == 0:
        return flat_list[0]
    if len(target_shape) == 1:
        return list(flat_list)

    def _build_recursive(iterator, shape):
        if len(shape) == 1:
            row = []
            for _ in range(shape[0]):
                row.append(next(iterator))
            return row
        dim = shape[0]
        sub_shape = shape[1:]
        res = []
        for _ in range(dim):
            res.append(_build_recursive(iterator, sub_shape))
        return res

    it = iter(flat_list)
    return _build_recursive(it, target_shape)


def _matrix_transpose(mat):
    rows = len(mat)
    cols = len(mat[0])
    transposed = []
    for c in range(cols):
        new_row = []
        for r in range(rows):
            new_row.append(mat[r][c])
        transposed.append(new_row)
    return transposed


def _raw_matmul(mat_a, mat_b):
    rows_a = len(mat_a)
    cols_a = len(mat_a[0])
    rows_b = len(mat_b)
    cols_b = len(mat_b[0])

    if cols_a != rows_b:
        raise ValueError("Cannot multiply shapes: (" + str(rows_a) + ", " + str(cols_a) + ") and (" + str(rows_b) + ", " + str(cols_b) + ")")

    result = []
    for i in range(rows_a):
        row_out = []
        for j in range(cols_b):
            dot_sum = 0.0
            for k in range(cols_a):
                dot_sum = dot_sum + mat_a[i][k] * mat_b[k][j]
            row_out.append(dot_sum)
        result.append(row_out)
    return result


class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _op=""):
        if isinstance(data, (int, float)):
            self.data = float(data)
            self.shape = ()
        elif isinstance(data, list):
            self.data = data
            s = []
            current = data
            while isinstance(current, list):
                length = len(current)
                s.append(length)
                if length > 0:
                    current = current[0]
                else:
                    break
            self.shape = tuple(s)
        else:
            raise TypeError("Unsupported data type for Tensor: " + str(type(data)))

        self.requires_grad = requires_grad and _grad_enabled
        self.grad = None
        self._backward = lambda: None
        if _grad_enabled:
            self._prev = set(_parents)
        else:
            self._prev = set()
        self._op = _op

    def __repr__(self):
        repr_str = "Tensor(" + str(self.data)
        if self.requires_grad:
            repr_str = repr_str + ", requires_grad=True"
        repr_str = repr_str + ")"
        return repr_str

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        req_grad = _grad_enabled and (self.requires_grad or other.requires_grad)

        if self.shape == () and other.shape == ():
            result_val = self.data + other.data
            out = Tensor(result_val, requires_grad=req_grad, _parents=(self, other), _op="+")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = 0.0
                        self.grad = self.grad + 1.0 * out.grad

                    if other.requires_grad:
                        if other.grad is None:
                            other.grad = 0.0
                        other.grad = other.grad + 1.0 * out.grad

                out._backward = _backward
            return out

        elif len(self.shape) == 1 and len(other.shape) == 1:
            new_data = []
            len_self = len(self.data)
            for idx in range(len_self):
                val_a = self.data[idx]
                val_b = other.data[idx]
                summed = val_a + val_b
                new_data.append(summed)

            out = Tensor(new_data, requires_grad=req_grad, _parents=(self, other), _op="+")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = [0.0] * len(out.grad)
                        for i in range(len(out.grad)):
                            self.grad[i] = self.grad[i] + out.grad[i]

                    if other.requires_grad:
                        if other.grad is None:
                            other.grad = [0.0] * len(out.grad)
                        for j in range(len(out.grad)):
                            other.grad[j] = other.grad[j] + out.grad[j]

                out._backward = _backward
            return out

        elif len(self.shape) == 2 and len(other.shape) == 1:
            rows = self.shape[0]
            cols = self.shape[1]
            if cols != other.shape[0]:
                raise ValueError("Cannot broadcast bias of shape " + str(other.shape) + " to matrix with cols " + str(cols))

            new_grid = []
            for r in range(rows):
                new_row = []
                for c in range(cols):
                    sum_val = self.data[r][c] + other.data[c]
                    new_row.append(sum_val)
                new_grid.append(new_row)

            out = Tensor(new_grid, requires_grad=req_grad, _parents=(self, other), _op="+")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = _zeros_like_shape(self.shape)
                        for r in range(rows):
                            for c in range(cols):
                                self.grad[r][c] = self.grad[r][c] + out.grad[r][c]

                    if other.requires_grad:
                        if other.grad is None:
                            other.grad = [0.0] * cols
                        for c in range(cols):
                            bias_grad_sum = 0.0
                            for r in range(rows):
                                bias_grad_sum = bias_grad_sum + out.grad[r][c]
                            other.grad[c] = other.grad[c] + bias_grad_sum

                out._backward = _backward
            return out

        elif self.shape == other.shape:
            flat_self = _flatten_list(self.data)
            flat_other = _flatten_list(other.data)
            flat_res = []
            for i in range(len(flat_self)):
                flat_res.append(flat_self[i] + flat_other[i])

            new_grid = _unflatten_to_shape(flat_res, self.shape)
            out = Tensor(new_grid, requires_grad=req_grad, _parents=(self, other), _op="+")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = _zeros_like_shape(self.shape)
                        flat_out = _flatten_list(out.grad)
                        flat_g = _flatten_list(self.grad)
                        for i in range(len(flat_g)):
                            flat_g[i] = flat_g[i] + flat_out[i]
                        self.grad = _unflatten_to_shape(flat_g, self.shape)

                    if other.requires_grad:
                        if other.grad is None:
                            other.grad = _zeros_like_shape(other.shape)
                        flat_out = _flatten_list(out.grad)
                        flat_g = _flatten_list(other.grad)
                        for i in range(len(flat_g)):
                            flat_g[i] = flat_g[i] + flat_out[i]
                        other.grad = _unflatten_to_shape(flat_g, other.shape)

                out._backward = _backward
            return out

        else:
            raise NotImplementedError("Addition not supported for shapes: " + str(self.shape) + " and " + str(other.shape))

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return self + (-other)

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        req_grad = _grad_enabled and (self.requires_grad or other.requires_grad)

        if self.shape == () and other.shape == ():
            result_val = self.data * other.data
            out = Tensor(result_val, requires_grad=req_grad, _parents=(self, other), _op="*")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = 0.0
                        self.grad = self.grad + other.data * out.grad

                    if other.requires_grad:
                        if other.grad is None:
                            other.grad = 0.0
                        other.grad = other.grad + self.data * out.grad

                out._backward = _backward
            return out

        elif other.shape == ():
            flat_self = _flatten_list(self.data)
            scalar_val = other.data
            flat_res = []
            for v in flat_self:
                flat_res.append(v * scalar_val)
            out_data = _unflatten_to_shape(flat_res, self.shape)
            out = Tensor(out_data, requires_grad=req_grad, _parents=(self, other), _op="*")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = _zeros_like_shape(self.shape)
                        flat_og = _flatten_list(out.grad)
                        flat_g = _flatten_list(self.grad)
                        for i in range(len(flat_g)):
                            flat_g[i] = flat_g[i] + flat_og[i] * scalar_val
                        self.grad = _unflatten_to_shape(flat_g, self.shape)

                    if other.requires_grad:
                        if other.grad is None:
                            other.grad = 0.0
                        flat_og = _flatten_list(out.grad)
                        flat_x = _flatten_list(self.data)
                        s = 0.0
                        for i in range(len(flat_x)):
                            s = s + flat_x[i] * flat_og[i]
                        other.grad = other.grad + s

                out._backward = _backward
            return out

        elif self.shape == other.shape:
            flat_self = _flatten_list(self.data)
            flat_other = _flatten_list(other.data)
            flat_res = []
            for i in range(len(flat_self)):
                flat_res.append(flat_self[i] * flat_other[i])

            out_data = _unflatten_to_shape(flat_res, self.shape)
            out = Tensor(out_data, requires_grad=req_grad, _parents=(self, other), _op="*")

            if req_grad:
                def _backward():
                    flat_og = _flatten_list(out.grad)
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = _zeros_like_shape(self.shape)
                        flat_g = _flatten_list(self.grad)
                        for i in range(len(flat_g)):
                            flat_g[i] = flat_g[i] + flat_other[i] * flat_og[i]
                        self.grad = _unflatten_to_shape(flat_g, self.shape)

                    if other.requires_grad:
                        if other.grad is None:
                            other.grad = _zeros_like_shape(other.shape)
                        flat_g = _flatten_list(other.grad)
                        for i in range(len(flat_g)):
                            flat_g[i] = flat_g[i] + flat_self[i] * flat_og[i]
                        other.grad = _unflatten_to_shape(flat_g, other.shape)

                out._backward = _backward
            return out

        else:
            raise NotImplementedError("Multiplication not implemented for shapes: " + str(self.shape) + " and " + str(other.shape))

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("Power must be an int or float, got " + str(type(power)))

        req_grad = _grad_enabled and self.requires_grad

        def _elem_pow(val):
            return val ** power

        def _elem_pow_grad(val):
            return power * (val ** (power - 1))

        def _apply_pow(data):
            if isinstance(data, (int, float)):
                return _elem_pow(data)
            return [_apply_pow(x) for x in data]

        out_data = _apply_pow(self.data)
        out = Tensor(out_data, requires_grad=req_grad, _parents=(self,), _op="**" + str(power))

        if req_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = _zeros_like_shape(self.shape)

                    def _accum_pow_grad(self_data, self_grad, out_grad):
                        if isinstance(self_data, (int, float)):
                            local_derivative = _elem_pow_grad(self_data)
                            return self_grad + local_derivative * out_grad
                        for idx in range(len(self_data)):
                            if isinstance(self_data[idx], list):
                                _accum_pow_grad(self_data[idx], self_grad[idx], out_grad[idx])
                            else:
                                local_derivative = _elem_pow_grad(self_data[idx])
                                self_grad[idx] = self_grad[idx] + local_derivative * out_grad[idx]

                    if self.shape == ():
                        self.grad = _accum_pow_grad(self.data, self.grad, out.grad)
                    else:
                        _accum_pow_grad(self.data, self.grad, out.grad)

            out._backward = _backward
        return out

    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

        if len(self.shape) != 2 or len(other.shape) != 2:
            raise ValueError("matmul currently only supports 2D matrices, got " + str(self.shape) + " and " + str(other.shape))

        req_grad = _grad_enabled and (self.requires_grad or other.requires_grad)
        res_data = _raw_matmul(self.data, other.data)
        out = Tensor(
            res_data,
            requires_grad=req_grad,
            _parents=(self, other),
            _op="matmul",
        )

        if req_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = _zeros_like_shape(self.shape)
                    other_t = _matrix_transpose(other.data)
                    grad_self = _raw_matmul(out.grad, other_t)
                    for r in range(len(self.grad)):
                        for c in range(len(self.grad[0])):
                            self.grad[r][c] = self.grad[r][c] + grad_self[r][c]

                if other.requires_grad:
                    if other.grad is None:
                        other.grad = _zeros_like_shape(other.shape)
                    self_t = _matrix_transpose(self.data)
                    grad_other = _raw_matmul(self_t, out.grad)
                    for r in range(len(other.grad)):
                        for c in range(len(other.grad[0])):
                            other.grad[r][c] = other.grad[r][c] + grad_other[r][c]

            out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            target_shape = tuple(shape[0])
        else:
            target_shape = tuple(shape)

        flat_elements = _flatten_list(self.data) if isinstance(self.data, list) else [self.data]
        total_elements = len(flat_elements)

        inferred = []
        neg_idx = -1
        known_product = 1

        for i, dim in enumerate(target_shape):
            if dim == -1:
                if neg_idx != -1:
                    raise ValueError("Can only specify one unknown dimension in reshape")
                neg_idx = i
                inferred.append(None)
            elif dim > 0:
                known_product = known_product * dim
                inferred.append(dim)
            else:
                raise ValueError("Invalid dimension size: " + str(dim))

        if neg_idx != -1:
            if known_product == 0 or total_elements % known_product != 0:
                raise ValueError("Cannot infer dimension for total elements " + str(total_elements) + " with shape " + str(target_shape))
            inferred[neg_idx] = total_elements // known_product

        final_shape = tuple(inferred)

        check_prod = 1
        for dim in final_shape:
            check_prod = check_prod * dim
        if check_prod != total_elements:
            raise ValueError("Total elements " + str(total_elements) + " does not match target shape " + str(final_shape))

        reshaped_data = _unflatten_to_shape(flat_elements, final_shape)
        req_grad = _grad_enabled and self.requires_grad

        out = Tensor(reshaped_data, requires_grad=req_grad, _parents=(self,), _op="reshape")

        if req_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = _zeros_like_shape(self.shape)

                    flat_out_grad = _flatten_list(out.grad) if isinstance(out.grad, list) else [out.grad]
                    restored_grad = _unflatten_to_shape(flat_out_grad, self.shape)

                    flat_self_g = _flatten_list(self.grad) if isinstance(self.grad, list) else [self.grad]
                    for i in range(len(flat_self_g)):
                        flat_self_g[i] = flat_self_g[i] + flat_out_grad[i]

                    if self.shape == ():
                        self.grad = flat_self_g[0]
                    else:
                        self.grad = _unflatten_to_shape(flat_self_g, self.shape)

            out._backward = _backward
        return out

    def transpose(self, dim0=0, dim1=1):
        if len(self.shape) != 2:
            raise NotImplementedError("transpose currently only implemented for 2D tensors, got shape " + str(self.shape))

        if {dim0, dim1} != {0, 1}:
            raise ValueError("For 2D tensor, transpose dimensions must be 0 and 1, got " + str((dim0, dim1)))

        transposed_data = _matrix_transpose(self.data)
        req_grad = _grad_enabled and self.requires_grad

        out = Tensor(transposed_data, requires_grad=req_grad, _parents=(self,), _op="transpose")

        if req_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = _zeros_like_shape(self.shape)
                    grad_restored = _matrix_transpose(out.grad)
                    for r in range(len(self.grad)):
                        for c in range(len(self.grad[0])):
                            self.grad[r][c] = self.grad[r][c] + grad_restored[r][c]

            out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose(0, 1)

    def relu(self):
        req_grad = _grad_enabled and self.requires_grad

        def _apply_relu(data):
            if isinstance(data, (int, float)):
                return data if data > 0.0 else 0.0
            return [_apply_relu(item) for item in data]

        out_data = _apply_relu(self.data)
        out = Tensor(out_data, requires_grad=req_grad, _parents=(self,), _op="relu")

        if req_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = _zeros_like_shape(self.shape)

                    def _accum_relu_grad(self_data, self_grad, out_grad):
                        if isinstance(self_data, (int, float)):
                            deriv = 1.0 if self_data > 0.0 else 0.0
                            return self_grad + deriv * out_grad
                        for i in range(len(self_data)):
                            if isinstance(self_data[i], list):
                                _accum_relu_grad(self_data[i], self_grad[i], out_grad[i])
                            else:
                                deriv = 1.0 if self_data[i] > 0.0 else 0.0
                                self_grad[i] = self_grad[i] + deriv * out_grad[i]

                    if self.shape == ():
                        self.grad = _accum_relu_grad(self.data, self.grad, out.grad)
                    else:
                        _accum_relu_grad(self.data, self.grad, out.grad)

            out._backward = _backward
        return out

    def sigmoid(self):
        def _calc_sigmoid(x):
            if x < -500.0:
                return 0.0
            if x > 500.0:
                return 1.0
            return 1.0 / (1.0 + math.exp(-x))

        req_grad = _grad_enabled and self.requires_grad

        def _apply_sig(data):
            if isinstance(data, (int, float)):
                return _calc_sigmoid(data)
            return [_apply_sig(item) for item in data]

        out_data = _apply_sig(self.data)
        out = Tensor(out_data, requires_grad=req_grad, _parents=(self,), _op="sigmoid")

        if req_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = _zeros_like_shape(self.shape)

                    def _accum_sig_grad(out_data_node, self_grad, out_grad):
                        if isinstance(out_data_node, (int, float)):
                            deriv = out_data_node * (1.0 - out_data_node)
                            return self_grad + deriv * out_grad
                        for i in range(len(out_data_node)):
                            if isinstance(out_data_node[i], list):
                                _accum_sig_grad(out_data_node[i], self_grad[i], out_grad[i])
                            else:
                                deriv = out_data_node[i] * (1.0 - out_data_node[i])
                                self_grad[i] = self_grad[i] + deriv * out_grad[i]

                    if self.shape == ():
                        self.grad = _accum_sig_grad(out.data, self.grad, out.grad)
                    else:
                        _accum_sig_grad(out.data, self.grad, out.grad)

            out._backward = _backward
        return out

    def tanh(self):
        def _calc_tanh(x):
            if x < -50.0:
                return -1.0
            if x > 50.0:
                return 1.0
            return math.tanh(x)

        req_grad = _grad_enabled and self.requires_grad

        def _apply_tanh(data):
            if isinstance(data, (int, float)):
                return _calc_tanh(data)
            return [_apply_tanh(item) for item in data]

        out_data = _apply_tanh(self.data)
        out = Tensor(out_data, requires_grad=req_grad, _parents=(self,), _op="tanh")

        if req_grad:
            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = _zeros_like_shape(self.shape)

                    def _accum_tanh_grad(out_data_node, self_grad, out_grad):
                        if isinstance(out_data_node, (int, float)):
                            deriv = 1.0 - (out_data_node * out_data_node)
                            return self_grad + deriv * out_grad
                        for i in range(len(out_data_node)):
                            if isinstance(out_data_node[i], list):
                                _accum_tanh_grad(out_data_node[i], self_grad[i], out_grad[i])
                            else:
                                deriv = 1.0 - (out_data_node[i] * out_data_node[i])
                                self_grad[i] = self_grad[i] + deriv * out_grad[i]

                    if self.shape == ():
                        self.grad = _accum_tanh_grad(out.data, self.grad, out.grad)
                    else:
                        _accum_tanh_grad(out.data, self.grad, out.grad)

            out._backward = _backward
        return out

    def sum(self):
        flat_items = _flatten_list(self.data) if isinstance(self.data, list) else [self.data]
        total = 0.0
        for item in flat_items:
            total = total + item

        req_grad = _grad_enabled and self.requires_grad
        out = Tensor(total, requires_grad=req_grad, _parents=(self,), _op="sum")

        if req_grad:
            def _backward():
                if self.requires_grad:
                    if self.shape == ():
                        if self.grad is None:
                            self.grad = 0.0
                        self.grad = self.grad + 1.0 * out.grad
                    else:
                        if self.grad is None:
                            self.grad = _zeros_like_shape(self.shape)

                        def _accumulate_ones(g_target, g_val):
                            if isinstance(g_target, list):
                                for idx in range(len(g_target)):
                                    if isinstance(g_target[idx], list):
                                        _accumulate_ones(g_target[idx], g_val)
                                    else:
                                        g_target[idx] = g_target[idx] + g_val

                        _accumulate_ones(self.grad, 1.0 * out.grad)

            out._backward = _backward
        return out

    def mean(self):
        total_elements = 1
        for dim in self.shape:
            total_elements = total_elements * dim
        if total_elements == 0:
            total_elements = 1

        summed = self.sum()
        scale = 1.0 / float(total_elements)
        return summed * scale

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        if self.shape == ():
            self.grad = 1.0
        else:
            self.grad = _zeros_like_shape(self.shape)

            def _fill_ones(g_target):
                if isinstance(g_target, list):
                    for idx in range(len(g_target)):
                        if isinstance(g_target[idx], list):
                            _fill_ones(g_target[idx])
                        else:
                            g_target[idx] = 1.0

            _fill_ones(self.grad)

        reversed_nodes = list(reversed(topo))
        for node in reversed_nodes:
            node._backward()