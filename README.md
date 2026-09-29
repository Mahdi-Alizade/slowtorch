# SlowTorch

A minimalist, educational deep learning library built from the ground up in 100% pure Python without third-party mathematical or autograd dependencies.

## Key Features

- **Dynamic Autograd Engine:** Reverse-mode automatic differentiation with DAG topological sorting.
- **Custom Tensor:** Multi-dimensional operations, matrix multiplications (`matmul`), and broadcasting.
- **Neural Network Primitives:** `Module`, `Parameter`, `Linear`, `ReLU`, `Sigmoid`, `Softmax`, `Dropout`, and `Sequential`.
- **Loss Functions:** `MSELoss` and numerically stable `CrossEntropyLoss`.
- **First-class Optimizers:** Vanilla `SGD` and adaptive `Adam`.
- **Model Checkpointing:** Native serialization with `state_dict`, `load_state_dict`, and JSON export.

## Installation

Activate your virtual environment and install in editable mode:

```bash
pip install -e .
Running Tests
Bash
pytest tests/