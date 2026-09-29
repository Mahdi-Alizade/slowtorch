class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.base_lr = float(optimizer.lr)
        self.last_epoch = int(last_epoch)
        self.step()

    def get_lr(self):
        raise NotImplementedError("Subclasses of _LRScheduler must implement get_lr")

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch = self.last_epoch + 1
        else:
            self.last_epoch = int(epoch)

        new_lr = self.get_lr()
        self.optimizer.lr = new_lr

    def state_dict(self):
        return {
            "base_lr": self.base_lr,
            "last_epoch": self.last_epoch
        }

    def load_state_dict(self, state_dict):
        if "base_lr" in state_dict:
            self.base_lr = float(state_dict["base_lr"])
        if "last_epoch" in state_dict:
            self.last_epoch = int(state_dict["last_epoch"])
        self.optimizer.lr = self.get_lr()


class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        if step_size <= 0:
            raise ValueError("step_size must be greater than 0, got " + str(step_size))
        self.step_size = int(step_size)
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr

        exponent = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma ** exponent)

    def state_dict(self):
        state = super().state_dict()
        state["step_size"] = self.step_size
        state["gamma"] = self.gamma
        return state

    def load_state_dict(self, state_dict):
        if "step_size" in state_dict:
            self.step_size = int(state_dict["step_size"])
        if "gamma" in state_dict:
            self.gamma = float(state_dict["gamma"])
        super().load_state_dict(state_dict)