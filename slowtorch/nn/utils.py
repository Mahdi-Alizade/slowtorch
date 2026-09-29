import math
from slowtorch.tensor import Tensor


def _sum_squared_grads(grad_data):
    if grad_data is None:
        return 0.0

    if isinstance(grad_data, (int, float)):
        return float(grad_data * grad_data)

    total = 0.0
    if isinstance(grad_data, list):
        for item in grad_data:
            total = total + _sum_squared_grads(item)
    return total


def _scale_grad_in_place(grad_data, scale_factor):
    if isinstance(grad_data, (int, float)):
        return grad_data * scale_factor

    if isinstance(grad_data, list):
        for idx in range(len(grad_data)):
            grad_data[idx] = _scale_grad_in_place(grad_data[idx], scale_factor)
        return grad_data

    return grad_data


def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
    if isinstance(parameters, Tensor):
        parameters = [parameters]

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if norm_type != 2.0:
        raise NotImplementedError("clip_grad_norm_ currently only supports L2 norm (norm_type=2.0)")

    total_sq_sum = 0.0
    valid_params = []

    for p in parameters:
        if p.grad is not None:
            valid_params.append(p)
            sq_sum = _sum_squared_grads(p.grad)
            total_sq_sum = total_sq_sum + sq_sum

    total_norm = math.sqrt(total_sq_sum)

    if error_if_nonfinite:
        if math.isnan(total_norm) or math.isinf(total_norm):
            raise RuntimeError("The total norm of order 2.0 is non-finite: " + str(total_norm))

    # Scale gradients if total norm exceeds threshold
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in valid_params:
            if p.shape == ():
                p.grad = p.grad * clip_coef
            else:
                _scale_grad_in_place(p.grad, clip_coef)

    return total_norm