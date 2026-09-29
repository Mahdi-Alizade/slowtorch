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

        # Exponent = floor(epoch / step_size)
        exponent = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma ** exponent)