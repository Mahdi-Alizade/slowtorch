import math


def _deep_copy_nested_list(data):
    if isinstance(data, (int, float)):
        return float(data)
    if isinstance(data, list):
        res = []
        for item in data:
            res.append(_deep_copy_nested_list(item))
        return res
    return data


def _create_zeros_like(data):
    if isinstance(data, (int, float)):
        return 0.0
    if isinstance(data, list):
        out = []
        for item in data:
            out.append(_create_zeros_like(item))
        return out
    return 0.0


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

            if p.shape == ():
                p.data = p.data - self.lr * p.grad

            elif len(p.shape) == 1:
                for i in range(len(p.data)):
                    p.data[i] = p.data[i] - self.lr * p.grad[i]

            elif len(p.shape) == 2:
                rows = p.shape[0]
                cols = p.shape[1]
                for r in range(rows):
                    for c in range(cols):
                        p.data[r][c] = p.data[r][c] - self.lr * p.grad[r][c]

    def state_dict(self):
        return {
            "type": "SGD",
            "lr": self.lr
        }

    def load_state_dict(self, state_dict):
        if "lr" in state_dict:
            self.lr = float(state_dict["lr"])


class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = float(lr)
        self.beta1 = float(betas[0])
        self.beta2 = float(betas[1])
        self.eps = float(eps)

        self.t = 0
        self.m = []
        self.v = []

        for p in self.params:
            self.m.append(_create_zeros_like(p.data))
            self.v.append(_create_zeros_like(p.data))

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
        self.t = self.t + 1

        bias_correction1 = 1.0 - (self.beta1 ** self.t)
        bias_correction2 = 1.0 - (self.beta2 ** self.t)

        for idx in range(len(self.params)):
            p = self.params[idx]
            if p.grad is None:
                continue

            if p.shape == ():
                g = p.grad
                self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * g
                self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * (g * g)

                m_hat = self.m[idx] / bias_correction1
                v_hat = self.v[idx] / bias_correction2

                step_size = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
                p.data = p.data - step_size

            elif len(p.shape) == 1:
                for i in range(len(p.data)):
                    g = p.grad[i]
                    self.m[idx][i] = self.beta1 * self.m[idx][i] + (1.0 - self.beta1) * g
                    self.v[idx][i] = self.beta2 * self.v[idx][i] + (1.0 - self.beta2) * (g * g)

                    m_hat = self.m[idx][i] / bias_correction1
                    v_hat = self.v[idx][i] / bias_correction2

                    step_size = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
                    p.data[i] = p.data[i] - step_size

            elif len(p.shape) == 2:
                rows = p.shape[0]
                cols = p.shape[1]
                for r in range(rows):
                    for c in range(cols):
                        g = p.grad[r][c]
                        self.m[idx][r][c] = self.beta1 * self.m[idx][r][c] + (1.0 - self.beta1) * g
                        self.v[idx][r][c] = self.beta2 * self.v[idx][r][c] + (1.0 - self.beta2) * (g * g)

                        m_hat = self.m[idx][r][c] / bias_correction1
                        v_hat = self.v[idx][r][c] / bias_correction2

                        step_size = self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
                        p.data[r][c] = p.data[r][c] - step_size

    def state_dict(self):
        m_copies = []
        for item in self.m:
            m_copies.append(_deep_copy_nested_list(item))

        v_copies = []
        for item in self.v:
            v_copies.append(_deep_copy_nested_list(item))

        return {
            "type": "Adam",
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "t": self.t,
            "m": m_copies,
            "v": v_copies,
        }

    def load_state_dict(self, state_dict):
        self.lr = float(state_dict.get("lr", self.lr))
        self.beta1 = float(state_dict.get("beta1", self.beta1))
        self.beta2 = float(state_dict.get("beta2", self.beta2))
        self.eps = float(state_dict.get("eps", self.eps))
        self.t = int(state_dict.get("t", self.t))

        if "m" in state_dict:
            self.m = []
            for item in state_dict["m"]:
                self.m.append(_deep_copy_nested_list(item))

        if "v" in state_dict:
            self.v = []
            for item in state_dict["v"]:
                self.v.append(_deep_copy_nested_list(item))