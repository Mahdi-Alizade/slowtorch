from slowtorch.tensor import Tensor, no_grad, is_grad_enabled
from slowtorch.nn import Linear


def test_no_grad_context_manager():
    a = Tensor(2.0, requires_grad=True)
    b = Tensor(3.0, requires_grad=True)

    assert is_grad_enabled() is True

    with no_grad():
        assert is_grad_enabled() is False
        c = a * b
        d = c + 5.0
        # Output tensors should NOT require gradients inside no_grad
        assert d.requires_grad is False
        assert len(d._prev) == 0

    # Ensure grad_enabled reverts back to True
    assert is_grad_enabled() is True

    # Normal computation outside no_grad tracks graph
    e = a * b
    assert e.requires_grad is True
    assert len(e._prev) == 2


def test_no_grad_decorator():
    @no_grad()
    def evaluate_model(model, x):
        return model(x)

    model = Linear(2, 3, bias=True)
    x = Tensor([[1.0, 2.0]], requires_grad=True)

    assert is_grad_enabled() is True
    out = evaluate_model(model, x)

    assert out.requires_grad is False
    assert len(out._prev) == 0
    assert is_grad_enabled() is True