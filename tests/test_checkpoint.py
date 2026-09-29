import os
from slowtorch.tensor import Tensor
from slowtorch.nn import Linear
from slowtorch.optim import Adam, StepLR
from slowtorch.checkpoint import save_checkpoint, load_checkpoint


def test_full_checkpoint_cycle(tmp_path):
    ckpt_file = str(tmp_path / "checkpoint.json")

    # Step 1: Initialize model, optimizer, scheduler
    model = Linear(2, 1, bias=True)
    model.weight.data = [[1.5], [-2.0]]
    model.bias.data = [0.25]

    optimizer = Adam(model.parameters(), lr=0.05)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Simulate 3 training steps
    x = Tensor([[1.0, 2.0]], requires_grad=False)
    for _ in range(3):
        optimizer.zero_grad()
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Save state
    save_checkpoint(
        ckpt_file,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        extra_info={"val_loss": 0.42}
    )

    assert os.path.exists(ckpt_file)

    # Step 2: Initialize fresh instances with different values
    new_model = Linear(2, 1, bias=True)
    new_optimizer = Adam(new_model.parameters(), lr=0.99)
    new_scheduler = StepLR(new_optimizer, step_size=5, gamma=0.5)

    # Load checkpoint
    ckpt_data = load_checkpoint(
        ckpt_file,
        model=new_model,
        optimizer=new_optimizer,
        scheduler=new_scheduler
    )

    # Verifications
    assert ckpt_data["epoch"] == 3
    assert ckpt_data["extra_info"]["val_loss"] == 0.42

    # Model weights must match exactly
    assert new_model.weight.data == model.weight.data
    assert new_model.bias.data == model.bias.data

    # Optimizer moments and step count must match
    assert new_optimizer.t == optimizer.t
    assert new_optimizer.lr == optimizer.lr
    assert new_optimizer.m == optimizer.m
    assert new_optimizer.v == optimizer.v

    # Scheduler epoch tracking must match
    assert new_scheduler.last_epoch == scheduler.last_epoch