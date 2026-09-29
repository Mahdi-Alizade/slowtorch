import random
from slowtorch.tensor import Tensor
from slowtorch.data import Dataset, TensorDataset, DataLoader
from slowtorch.nn import Linear, MSELoss
from slowtorch.optim import SGD


def test_tensor_dataset_and_length():
    x = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = Tensor([[0.0], [1.0], [0.0]])

    ds = TensorDataset(x, y)
    assert len(ds) == 3

    sample_x, sample_y = ds[1]
    assert sample_x == [3.0, 4.0]
    assert sample_y == [1.0]


def test_dataloader_batching_and_drop_last():
    # 5 samples with batch_size 2
    x = Tensor([[float(i)] for i in range(5)])
    y = Tensor([[float(i * 10)] for i in range(5)])

    dataset = TensorDataset(x, y)

    # 1. drop_last = False -> 3 batches (2, 2, 1)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=False)
    assert len(loader) == 3

    batches = list(loader)
    assert len(batches) == 3
    assert batches[0][0].shape == (2, 1)
    assert batches[1][0].shape == (2, 1)
    assert batches[2][0].shape == (1, 1)

    # 2. drop_last = True -> 2 batches (2, 2)
    loader_drop = DataLoader(dataset, batch_size=2, shuffle=False, drop_last=True)
    assert len(loader_drop) == 2
    batches_drop = list(loader_drop)
    assert len(batches_drop) == 2


def test_dataloader_shuffle_changes_order():
    random.seed(42)
    x = Tensor([[float(i)] for i in range(20)])
    ds = TensorDataset(x)

    loader = DataLoader(ds, batch_size=20, shuffle=True)
    batch = next(iter(loader))

    extracted_order = [row[0] for row in batch.data]
    sequential_order = [float(i) for i in range(20)]

    # Shuffled order must not be identical to sequential order
    assert extracted_order != sequential_order
    assert sorted(extracted_order) == sequential_order


def test_dataloader_training_loop_integration():
    x_data = [[1.0], [2.0], [3.0], [4.0]]
    y_data = [[2.0], [4.0], [6.0], [8.0]]

    x = Tensor(x_data)
    y = Tensor(y_data)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = Linear(1, 1, bias=False)
    model.weight.data = [[0.1]]

    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    initial_loss = criterion(model(x), y).data

    for _ in range(20):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    final_loss = criterion(model(x), y).data
    assert final_loss < initial_loss