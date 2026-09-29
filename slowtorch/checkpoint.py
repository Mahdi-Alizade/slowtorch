import json


def save_checkpoint(filepath, model=None, optimizer=None, scheduler=None, epoch=None, extra_info=None):
    checkpoint = {}

    if model is not None:
        checkpoint["model_state_dict"] = model.state_dict()

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = int(epoch)

    if extra_info is not None:
        checkpoint["extra_info"] = extra_info

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(filepath, model=None, optimizer=None, scheduler=None):
    with open(filepath, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    if model is not None and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint