# D:\Mahdi Alizade\Projects\slowtorch\slowtorch\tensor.py


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
        else:
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

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)

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
        else:
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