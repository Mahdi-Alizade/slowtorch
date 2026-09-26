# D:\Mahdi Alizade\Projects\slowtorch\slowtorch\tensor.py

class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _op=""):
        # We check whether data is scalar or list
        if isinstance(data, (int, float)):
            self.data = float(data)
            self.shape = ()
        elif isinstance(data, list):
            self.data = data
            # Calculate shape manually for multi-dimensional nested lists
            current = data
            s = []
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
        # Ensure 'other' is a Tensor instance
        if not isinstance(other, Tensor):
            other = Tensor(other)

        # For now, let's implement element-wise addition for scalar/1D values
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
            # Handle 1D vector addition
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
            # Element-wise list multiplication
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

    def backward(self):
        # Topological sort to traverse the computational graph
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        # Base gradient initialization
        if self.shape == ():
            self.grad = 1.0
        else:
            self.grad = [1.0] * len(self.data)

        # Traverse backwards through topological order
        reversed_nodes = list(reversed(topo))
        for node in reversed_nodes:
            node._backward()