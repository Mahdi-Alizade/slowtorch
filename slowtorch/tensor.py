import math


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

        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set(_parents)
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

        if self.shape == () and other.shape == ():
            result_val = self.data + other.data
            out = Tensor(result_val, requires_grad=(self.requires_grad or other.requires_grad), _parents=(self, other), _op="+")

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

            out = Tensor(new_data, requires_grad=(self.requires_grad or other.requires_grad), _parents=(self, other), _op="+")

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

            out = Tensor(new_grid, requires_grad=(self.requires_grad or other.requires_grad), _parents=(self, other), _op="+")

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

            out = Tensor(new_grid, requires_grad=(self.requires_grad or other.requires_grad), _parents=(self, other), _op="+")

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

        # Scalar * Scalar
        if self.shape == () and other.shape == ():
            result_val = self.data * other.data
            out = Tensor(result_val, requires_grad=(self.requires_grad or other.requires_grad), _parents=(self, other), _op="*")

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

        # 2D Matrix * Scalar
        elif len(self.shape) == 2 and other.shape == ():
            new_data = []
            rows = self.shape[0]
            cols = self.shape[1]
            for r in range(rows):
                row = []
                for c in range(cols):
                    row.append(self.data[r][c] * other.data)
                new_data.append(row)

            out = Tensor(new_data, requires_grad=(self.requires_grad or other.requires_grad), _parents=(self, other), _op="*")

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

        # 1D Vector * 1D Vector (Element-wise)
        elif len(self.shape) == 1 and len(other.shape) == 1:
            new_data = []
            for i in range(len(self.data)):
                mult_val = self.data[i] * other.data[i]
                new_data.append(mult_val)

            out = Tensor(new_data, requires_grad=(self.requires_grad or other.requires_grad), _parents=(self, other), _op="*")

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

        # 2D Matrix * 2D Matrix (Element-wise)
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

            out = Tensor(new_grid, requires_grad=(self.requires_grad or other.requires_grad), _parents=(self, other), _op="*")

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

        # Scalar
        if self.shape == ():
            val = self.data ** power
            out = Tensor(val, requires_grad=self.requires_grad, _parents=(self,), _op="**" + str(power))

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = 0.0
                    local_grad = power * (self.data ** (power - 1))
                    self.grad = self.grad + local_grad * out.grad

            out._backward = _backward
            return out

        # 1D Vector
        elif len(self.shape) == 1:
            out_data = []
            for item in self.data:
                out_data.append(item ** power)

            out = Tensor(out_data, requires_grad=self.requires_grad, _parents=(self,), _op="**" + str(power))

            def _backward():
                if self.requires_grad:
                    if self.grad is None:
                        self.grad = [0.0] * len(self.data)
                    for i in range(len(self.data)):
                        local_grad = power * (self.data[i] ** (power - 1))
                        self.grad[i] = self.grad[i] + local_grad * out.grad[i]

            out._backward = _backward
            return out

        # 2D Matrix
        elif len(self.shape) == 2:
            rows = self.shape[0]
            cols = self.shape[1]
            out_data = []
            for r in range(rows):
                row_items = []
                for c in range(cols):
                    row_items.append(self.data[r][c] ** power)
                out_data.append(row_items)

            out = Tensor(out_data, requires_grad=self.requires_grad, _parents=(self,), _op="**" + str(power))

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

        res_data = _raw_matmul(self.data, other.data)
        out = Tensor(
            res_data,
            requires_grad=(self.requires_grad or other.requires_grad),
            _parents=(self, other),
            _op="matmul",
        )

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

    def relu(self):
        if self.shape == ():
            val = self.data if self.data > 0.0 else 0.0
            out = Tensor(val, requires_grad=self.requires_grad, _parents=(self,), _op="relu")

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

            out = Tensor(out_data, requires_grad=self.requires_grad, _parents=(self,), _op="relu")

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

            out = Tensor(out_data, requires_grad=self.requires_grad, _parents=(self,), _op="relu")

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

        if self.shape == ():
            sig_val = _calc_sigmoid(self.data)
            out = Tensor(sig_val, requires_grad=self.requires_grad, _parents=(self,), _op="sigmoid")

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

            out = Tensor(out_data, requires_grad=self.requires_grad, _parents=(self,), _op="sigmoid")

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

            out = Tensor(out_data, requires_grad=self.requires_grad, _parents=(self,), _op="sigmoid")

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

        out = Tensor(total, requires_grad=self.requires_grad, _parents=(self,), _op="sum")

        def _backward():
            if self.requires_grad:
                if self.shape == ():
                    if self.grad is None:
                        self.grad = 0.0
                    self.grad = self.grad + 1.0 * out.grad
                elif len(self.shape) == 1:
                    if self.grad is None:
                        self.grad = [0.0] * len(self.data)
                    for i in range(len(self.grad)):
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