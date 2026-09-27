class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = float(lr)

    def zero_grad(self):
        for p in self.params:
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

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue

            # Update scalar parameter
            if p.shape == ():
                p.data = p.data - self.lr * p.grad

            # Update 1D vector parameter (e.g. bias)
            elif len(p.shape) == 1:
                for i in range(len(p.data)):
                    p.data[i] = p.data[i] - self.lr * p.grad[i]

            # Update 2D matrix parameter (e.g. weight)
            elif len(p.shape) == 2:
                rows = p.shape[0]
                cols = p.shape[1]
                for r in range(rows):
                    for c in range(cols):
                        p.data[r][c] = p.data[r][c] - self.lr * p.grad[r][c]