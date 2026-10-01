import math
from slowtorch.tensor import Tensor, _zeros_like_shape


def relu(x):
    return x.relu()


def sigmoid(x):
    return x.sigmoid()


def softmax(x, dim=-1):
    if len(x.shape) != 2:
        raise NotImplementedError("softmax currently only supports 2D tensors (batch_size, num_classes)")

    rows = x.shape[0]
    cols = x.shape[1]
    probs = []

    for r in range(rows):
        max_val = x.data[r][0]
        for c in range(1, cols):
            if x.data[r][c] > max_val:
                max_val = x.data[r][c]

        exp_vals = []
        exp_sum = 0.0
        for c in range(cols):
            e = math.exp(x.data[r][c] - max_val)
            exp_vals.append(e)
            exp_sum = exp_sum + e

        row_prob = []
        for c in range(cols):
            row_prob.append(exp_vals[c] / exp_sum)
        probs.append(row_prob)

    out = Tensor(probs, requires_grad=x.requires_grad, _parents=(x,), _op="softmax")

    def _backward():
        if x.requires_grad:
            if x.grad is None:
                x.grad = _zeros_like_shape(x.shape)
            for r in range(rows):
                dot_grad_p = 0.0
                for c in range(cols):
                    dot_grad_p = dot_grad_p + out.grad[r][c] * probs[r][c]
                for c in range(cols):
                    local_grad = probs[r][c] * (out.grad[r][c] - dot_grad_p)
                    x.grad[r][c] = x.grad[r][c] + local_grad

    out._backward = _backward
    return out


def mse_loss(pred, target):
    diff = pred - target
    squared = diff ** 2
    return squared.mean()


def cross_entropy(logits, targets):
    if len(logits.shape) != 2:
        raise ValueError("Logits must be 2D tensor (batch_size, num_classes)")

    rows = logits.shape[0]
    cols = logits.shape[1]

    if isinstance(targets, Tensor):
        if len(targets.shape) == 2:
            target_indices = [int(targets.data[i][0]) for i in range(rows)]
        else:
            target_indices = [int(targets.data[i]) for i in range(rows)]
    elif isinstance(targets, list):
        target_indices = [int(idx) for idx in targets]
    else:
        raise TypeError("Unsupported targets format for cross_entropy")

    total_loss = 0.0
    probabilities = []

    for r in range(rows):
        max_val = logits.data[r][0]
        for c in range(1, cols):
            if logits.data[r][c] > max_val:
                max_val = logits.data[r][c]

        exp_vals = []
        sum_exp = 0.0
        for c in range(cols):
            shift = logits.data[r][c] - max_val
            e = math.exp(shift)
            exp_vals.append(e)
            sum_exp = sum_exp + e

        log_sum_exp = max_val + math.log(sum_exp)
        correct_class = target_indices[r]
        loss_item = log_sum_exp - logits.data[r][correct_class]
        total_loss = total_loss + loss_item

        row_prob = []
        for c in range(cols):
            row_prob.append(exp_vals[c] / sum_exp)
        probabilities.append(row_prob)

    mean_loss = total_loss / float(rows)
    out = Tensor(mean_loss, requires_grad=logits.requires_grad, _parents=(logits,), _op="cross_entropy")

    def _backward():
        if logits.requires_grad:
            if logits.grad is None:
                logits.grad = _zeros_like_shape(logits.shape)
            for r in range(rows):
                correct_class = target_indices[r]
                for c in range(cols):
                    p = probabilities[r][c]
                    if c == correct_class:
                        grad_val = (p - 1.0) / float(rows)
                    else:
                        grad_val = p / float(rows)
                    logits.grad[r][c] = logits.grad[r][c] + grad_val * out.grad

    out._backward = _backward
    return out


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    if not isinstance(x1, Tensor):
        x1 = Tensor(x1)
    if not isinstance(x2, Tensor):
        x2 = Tensor(x2)

    # 1D single vector comparison
    if len(x1.shape) == 1 and len(x2.shape) == 1:
        if len(x1.data) != len(x2.data):
            raise ValueError("Vector dimensions must match for cosine_similarity")
        n = len(x1.data)

        dot = 0.0
        norm1_sq = 0.0
        norm2_sq = 0.0
        for i in range(n):
            u = x1.data[i]
            v = x2.data[i]
            dot = dot + u * v
            norm1_sq = norm1_sq + u * u
            norm2_sq = norm2_sq + v * v

        norm1 = math.sqrt(norm1_sq)
        norm2 = math.sqrt(norm2_sq)
        denom = norm1 * norm2 + eps
        sim_val = dot / denom

        req_grad = x1.requires_grad or x2.requires_grad
        out = Tensor(sim_val, requires_grad=req_grad, _parents=(x1, x2), _op="cosine_similarity")

        def _backward():
            if x1.requires_grad:
                if x1.grad is None:
                    x1.grad = [0.0] * n
                for i in range(n):
                    # d/du = v / (||u|| * ||v||) - sim * u / (||u||^2)
                    grad_u = (x2.data[i] / denom) - (sim_val * x1.data[i] / (norm1_sq + eps))
                    x1.grad[i] = x1.grad[i] + grad_u * out.grad

            if x2.requires_grad:
                if x2.grad is None:
                    x2.grad = [0.0] * n
                for i in range(n):
                    grad_v = (x1.data[i] / denom) - (sim_val * x2.data[i] / (norm2_sq + eps))
                    x2.grad[i] = x2.grad[i] + grad_v * out.grad

        out._backward = _backward
        return out

    # 2D batch of vectors comparison (dim=1)
    elif len(x1.shape) == 2 and len(x2.shape) == 2:
        if x1.shape != x2.shape:
            raise ValueError("Input shapes must match for batch cosine_similarity: " + str(x1.shape) + " vs " + str(x2.shape))

        rows, cols = x1.shape
        sims = []
        denom_list = []
        norm1_sq_list = []
        norm2_sq_list = []

        for r in range(rows):
            dot = 0.0
            norm1_sq = 0.0
            norm2_sq = 0.0
            for c in range(cols):
                u = x1.data[r][c]
                v = x2.data[r][c]
                dot = dot + u * v
                norm1_sq = norm1_sq + u * u
                norm2_sq = norm2_sq + v * v

            norm1 = math.sqrt(norm1_sq)
            norm2 = math.sqrt(norm2_sq)
            denom = norm1 * norm2 + eps
            val = dot / denom

            sims.append(val)
            denom_list.append(denom)
            norm1_sq_list.append(norm1_sq)
            norm2_sq_list.append(norm2_sq)

        req_grad = x1.requires_grad or x2.requires_grad
        out = Tensor(sims, requires_grad=req_grad, _parents=(x1, x2), _op="cosine_similarity")

        def _backward():
            if x1.requires_grad:
                if x1.grad is None:
                    x1.grad = _zeros_like_shape(x1.shape)
                for r in range(rows):
                    sim_r = sims[r]
                    denom_r = denom_list[r]
                    n1_sq = norm1_sq_list[r]
                    og = out.grad[r]
                    for c in range(cols):
                        gu = (x2.data[r][c] / denom_r) - (sim_r * x1.data[r][c] / (n1_sq + eps))
                        x1.grad[r][c] = x1.grad[r][c] + gu * og

            if x2.requires_grad:
                if x2.grad is None:
                    x2.grad = _zeros_like_shape(x2.shape)
                for r in range(rows):
                    sim_r = sims[r]
                    denom_r = denom_list[r]
                    n2_sq = norm2_sq_list[r]
                    og = out.grad[r]
                    for c in range(cols):
                        gv = (x1.data[r][c] / denom_r) - (sim_r * x2.data[r][c] / (n2_sq + eps))
                        x2.grad[r][c] = x2.grad[r][c] + gv * og

        out._backward = _backward
        return out

    else:
        raise NotImplementedError("cosine_similarity currently supports 1D and 2D matching shapes")


def pairwise_distance(x1, x2, p=2.0, eps=1e-6):
    if not isinstance(x1, Tensor):
        x1 = Tensor(x1)
    if not isinstance(x2, Tensor):
        x2 = Tensor(x2)

    p = float(p)
    if p != 2.0:
        raise NotImplementedError("pairwise_distance currently only supports p=2.0 (Euclidean distance)")

    # 1D single vectors
    if len(x1.shape) == 1 and len(x2.shape) == 1:
        if len(x1.data) != len(x2.data):
            raise ValueError("Vector dimensions must match for pairwise_distance")
        n = len(x1.data)

        sq_sum = 0.0
        diffs = []
        for i in range(n):
            d = x1.data[i] - x2.data[i]
            diffs.append(d)
            sq_sum = sq_sum + d * d

        dist = math.sqrt(sq_sum + eps)
        req_grad = x1.requires_grad or x2.requires_grad
        out = Tensor(dist, requires_grad=req_grad, _parents=(x1, x2), _op="pairwise_distance")

        def _backward():
            inv_dist = 1.0 / dist
            if x1.requires_grad:
                if x1.grad is None:
                    x1.grad = [0.0] * n
                for i in range(n):
                    grad_u = (diffs[i] * inv_dist) * out.grad
                    x1.grad[i] = x1.grad[i] + grad_u

            if x2.requires_grad:
                if x2.grad is None:
                    x2.grad = [0.0] * n
                for i in range(n):
                    grad_v = (-diffs[i] * inv_dist) * out.grad
                    x2.grad[i] = x2.grad[i] + grad_v

        out._backward = _backward
        return out

    # 2D batch of vectors
    elif len(x1.shape) == 2 and len(x2.shape) == 2:
        if x1.shape != x2.shape:
            raise ValueError("Input shapes must match for pairwise_distance: " + str(x1.shape) + " vs " + str(x2.shape))

        rows, cols = x1.shape
        distances = []
        diffs_grid = []

        for r in range(rows):
            sq_sum = 0.0
            row_diffs = []
            for c in range(cols):
                d = x1.data[r][c] - x2.data[r][c]
                row_diffs.append(d)
                sq_sum = sq_sum + d * d
            dist = math.sqrt(sq_sum + eps)
            distances.append(dist)
            diffs_grid.append(row_diffs)

        req_grad = x1.requires_grad or x2.requires_grad
        out = Tensor(distances, requires_grad=req_grad, _parents=(x1, x2), _op="pairwise_distance")

        def _backward():
            if x1.requires_grad:
                if x1.grad is None:
                    x1.grad = _zeros_like_shape(x1.shape)
                for r in range(rows):
                    inv_dist = 1.0 / distances[r]
                    og = out.grad[r]
                    for c in range(cols):
                        gu = (diffs_grid[r][c] * inv_dist) * og
                        x1.grad[r][c] = x1.grad[r][c] + gu

            if x2.requires_grad:
                if x2.grad is None:
                    x2.grad = _zeros_like_shape(x2.shape)
                for r in range(rows):
                    inv_dist = 1.0 / distances[r]
                    og = out.grad[r]
                    for c in range(cols):
                        gv = (-diffs_grid[r][c] * inv_dist) * og
                        x2.grad[r][c] = x2.grad[r][c] + gv

        out._backward = _backward
        return out

    else:
        raise NotImplementedError("pairwise_distance currently supports 1D and 2D matching shapes")