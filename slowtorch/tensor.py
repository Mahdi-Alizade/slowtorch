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
    if len(shape) == 1:
        res = []
        for _ in range(shape[0]):
            res.append(0.0)
        return res
    if len(shape) == 2:
        rows = shape[0]
        cols = shape[1]
        grid = []
        for r in range(rows):
            row = []
            for c in range(cols):
                row.append(0.0)
            grid.append(row)
        return grid
    raise ValueError("Shapes higher than 2D not implemented yet: " + str(shape))


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
    if len(target_shape) == 2:
        rows, cols = target_shape
        out = []
        idx = 0
        for _ in range(rows):
            row = []
            for _ in range(cols):
                row.append(flat_list[idx])
                idx = idx + 1
            out.append(row)
        return out
    raise NotImplementedError("Shapes beyond 2D not supported: " + str(target_shape))


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

        elif len(self.shape) == 2 and len(other.shape) == 2:
            if self.shape != other.shape:
                raise ValueError("Shape mismatch for 2D addition: " + str(self.shape) + " vs " + str(other.shape))
            rows = self.shape[0]
            cols = self.shape[1]
            new_grid = []
            for r in range(rows):
                new_row = []
                for c in range(cols):
                    new_row.append(self.data[r][c] + other.data[r][c])
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
                            other.grad = _zeros_like_shape(other.shape)
                        for r in range(rows):
                            for c in range(cols):
                                other.grad[r][c] = other.grad[r][c] + out.grad[r][c]

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

        elif len(self.shape) == 2 and other.shape == ():
            new_data = []
            rows = self.shape[0]
            cols = self.shape[1]
            for r in range(rows):
                row = []
                for c in range(cols):
                    row.append(self.data[r][c] * other.data)
                new_data.append(row)

            out = Tensor(new_data, requires_grad=req_grad, _parents=(self, other), _op="*")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = _zeros_like_shape(self.shape)
                        for r in range(rows):
                            for c in range(cols):
                                self.grad[r][c] = self.grad[r][c] + other.data * out.grad[r][c]

                    if other.requires_grad:
                        if other.grad is None:
                            other.grad = 0.0
                        for r in range(rows):
                            for c in range(cols):
                                other.grad = other.grad + self.data[r][c] * out.grad[r][c]

                out._backward = _backward
            return out

        elif len(self.shape) == 1 and len(other.shape) == 1:
            new_data = []
            for i in range(len(self.data)):
                mult_val = self.data[i] * other.data[i]
                new_data.append(mult_val)

            out = Tensor(new_data, requires_grad=req_grad, _parents=(self, other), _op="*")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = [0.0] * len(self.data)
                        for i in range(len(self.data)):
                            self.grad[i] = self.grad[i] + other.data[i] * out.grad[i]

                    if other.requires_grad:
                        if other.grad is None:
                            other.grad = [0.0] * len(other.data)
                        for j in range(len(other.data)):
                            other.grad[j] = other.grad[j] + self.data[j] * out.grad[j]

                out._backward = _backward
            return out

        elif len(self.shape) == 2 and len(other.shape) == 2:
            if self.shape != other.shape:
                raise ValueError("Shape mismatch for element-wise multiplication: " + str(self.shape) + " vs " + str(other.shape))
            rows = self.shape[0]
            cols = self.shape[1]
            new_grid = []
            for r in range(rows):
                new_row = []
                for c in range(cols):
                    new_row.append(self.data[r][c] * other.data[r][c])
                new_grid.append(new_row)

            out = Tensor(new_grid, requires_grad=req_grad, _parents=(self, other), _op="*")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = _zeros_like_shape(self.shape)
                        for r in range(rows):
                            for c in range(cols):
                                self.grad[r][c] = self.grad[r][c] + other.data[r][c] * out.grad[r][c]

                    if other.requires_grad:
                        if other.grad is None:
                            other.grad = _zeros_like_shape(other.shape)
                        for r in range(rows):
                            for c in range(cols):
                                other.grad[r][c] = other.grad[r][c] + self.data[r][c] * out.grad[r][c]

                out._backward = _backward
            return out

        else:
            raise NotImplementedError("Multiplication not implemented for shapes: " + str(self.shape) + " and " + str(other.shape))

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("Power must be an int or float, got " + str(type(power)))

        req_grad = _grad_enabled and self.requires_grad

        if self.shape == ():
            val = self.data ** power
            out = Tensor(val, requires_grad=req_grad, _parents=(self,), _op="**" + str(power))

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = 0.0
                        local_grad = power * (self.data ** (power - 1))
                        self.grad = self.grad + local_grad * out.grad

                out._backward = _backward
            return out

        elif len(self.shape) == 1:
            out_data = []
            for item in self.data:
                out_data.append(item ** power)

            out = Tensor(out_data, requires_grad=req_grad, _parents=(self,), _op="**" + str(power))

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = [0.0] * len(self.data)
                        for i in range(len(self.data)):
                            local_grad = power * (self.data[i] ** (power - 1))
                            self.grad[i] = self.grad[i] + local_grad * out.grad[i]

                out._backward = _backward
            return out

        elif len(self.shape) == 2:
            rows = self.shape[0]
            cols = self.shape[1]
            out_data = []
            for r in range(rows):
                row_items = []
                for c in range(cols):
                    row_items.append(self.data[r][c] ** power)
                out_data.append(row_items)

            out = Tensor(out_data, requires_grad=req_grad, _parents=(self,), _op="**" + str(power))

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = _zeros_like_shape(self.shape)
                        for r in range(rows):
                            for c in range(cols):
                                local_grad = power * (self.data[r][c] ** (power - 1))
                                self.grad[r][c] = self.grad[r][c] + local_grad * out.grad[r][c]

                out._backward = _backward
            return out
        else:
            raise NotImplementedError("Power not implemented for shape: " + str(self.shape))

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

        # Handle negative dimension inference (e.g. -1)
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

                    # Flatten output grad and reshape it back to self.shape
                    flat_out_grad = _flatten_list(out.grad) if isinstance(out.grad, list) else [out.grad]
                    restored_grad = _unflatten_to_shape(flat_out_grad, self.shape)

                    if self.shape == ():
                        self.grad = self.grad + restored_grad
                    elif len(self.shape) == 1:
                        for i in range(len(self.grad)):
                            self.grad[i] = self.grad[i] + restored_grad[i]
                    elif len(self.shape) == 2:
                        for r in range(len(self.grad)):
                            for c in range(len(self.grad[0])):
                                self.grad[r][c] = self.grad[r][c] + restored_grad[r][c]

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
                    # Transposing out.grad aligns back to self.shape
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

        if self.shape == ():
            val = self.data if self.data > 0.0 else 0.0
            out = Tensor(val, requires_grad=req_grad, _parents=(self,), _op="relu")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = 0.0
                        local_derivative = 1.0 if self.data > 0.0 else 0.0
                        self.grad = self.grad + local_derivative * out.grad

                out._backward = _backward
            return out

        elif len(self.shape) == 1:
            out_data = []
            for item in self.data:
                if item > 0.0:
                    out_data.append(item)
                else:
                    out_data.append(0.0)

            out = Tensor(out_data, requires_grad=req_grad, _parents=(self,), _op="relu")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = [0.0] * len(self.data)
                        for i in range(len(self.data)):
                            local_derivative = 1.0 if self.data[i] > 0.0 else 0.0
                            self.grad[i] = self.grad[i] + local_derivative * out.grad[i]

                out._backward = _backward
            return out

        elif len(self.shape) == 2:
            out_data = []
            rows = self.shape[0]
            cols = self.shape[1]
            for r in range(rows):
                row_items = []
                for c in range(cols):
                    elem = self.data[r][c]
                    if elem > 0.0:
                        row_items.append(elem)
                    else:
                        row_items.append(0.0)
                out_data.append(row_items)

            out = Tensor(out_data, requires_grad=req_grad, _parents=(self,), _op="relu")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = _zeros_like_shape(self.shape)
                        for r in range(rows):
                            for c in range(cols):
                                local_derivative = 1.0 if self.data[r][c] > 0.0 else 0.0
                                self.grad[r][c] = self.grad[r][c] + local_derivative * out.grad[r][c]

                out._backward = _backward
            return out
        else:
            raise NotImplementedError("ReLU not implemented for shape: " + str(self.shape))

    def sigmoid(self):
        def _calc_sigmoid(x):
            if x < -500.0:
                return 0.0
            if x > 500.0:
                return 1.0
            return 1.0 / (1.0 + math.exp(-x))

        req_grad = _grad_enabled and self.requires_grad

        if self.shape == ():
            sig_val = _calc_sigmoid(self.data)
            out = Tensor(sig_val, requires_grad=req_grad, _parents=(self,), _op="sigmoid")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = 0.0
                        local_derivative = out.data * (1.0 - out.data)
                        self.grad = self.grad + local_derivative * out.grad

                out._backward = _backward
            return out

        elif len(self.shape) == 1:
            out_data = []
            for item in self.data:
                out_data.append(_calc_sigmoid(item))

            out = Tensor(out_data, requires_grad=req_grad, _parents=(self,), _op="sigmoid")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = [0.0] * len(self.data)
                        for i in range(len(self.data)):
                            local_derivative = out.data[i] * (1.0 - out.data[i])
                            self.grad[i] = self.grad[i] + local_derivative * out.grad[i]

                out._backward = _backward
            return out

        elif len(self.shape) == 2:
            out_data = []
            rows = self.shape[0]
            cols = self.shape[1]
            for r in range(rows):
                row_items = []
                for c in range(cols):
                    elem = self.data[r][c]
                    row_items.append(_calc_sigmoid(elem))
                out_data.append(row_items)

            out = Tensor(out_data, requires_grad=req_grad, _parents=(self,), _op="sigmoid")

            if req_grad:
                def _backward():
                    if self.requires_grad:
                        if self.grad is None:
                            self.grad = _zeros_like_shape(self.shape)
                        for r in range(rows):
                            for c in range(cols):
                                local_derivative = out.data[r][c] * (1.0 - out.data[r][c])
                                self.grad[r][c] = self.grad[r][c] + local_derivative * out.grad[r][c]

                out._backward = _backward
            return out
        else:
            raise NotImplementedError("Sigmoid not implemented for shape: " + str(self.shape))

    def sum(self):
        total = 0.0
        if self.shape == ():
            total = self.data
        elif len(self.shape) == 1:
            for item in self.data:
                total = total + item
        elif len(self.shape) == 2:
            for row in self.data:
                for item in row:
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
                    elif len(self.shape) == 1:
                        if self.grad is None:
                            self.grad = [0.0] * len(self.data)
                        for i in range(len(self.data)):
                            self.grad[i] = self.grad[i] + 1.0 * out.grad
                    elif len(self.shape) == 2:
                        if self.grad is None:
                            self.grad = _zeros_like_shape(self.shape)
                        for r in range(len(self.grad)):
                            for c in range(len(self.grad[0])):
                                self.grad[r][c] = self.grad[r][c] + 1.0 * out.grad

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
        elif len(self.shape) == 1:
            self.grad = [1.0] * len(self.data)
        elif len(self.shape) == 2:
            rows = len(self.data)
            cols = len(self.data[0])
            self.grad = []
            for _ in range(rows):
                row = []
                for _ in range(cols):
                    row.append(1.0)
                self.grad.append(row)

        reversed_nodes = list(reversed(topo))
        for node in reversed_nodes:
            node._backward()