import random
from slowtorch.tensor import Tensor


class Dataset:
    def __len__(self):
        raise NotImplementedError("Subclasses of Dataset must implement __len__")

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses of Dataset must implement __getitem__")


class TensorDataset(Dataset):
    def __init__(self, *tensors):
        if len(tensors) == 0:
            raise ValueError("TensorDataset requires at least one tensor")

        first_len = len(tensors[0].data)
        for idx, t in enumerate(tensors):
            if not isinstance(t, Tensor):
                raise TypeError("All items in TensorDataset must be Tensor instances, got " + str(type(t)))
            if len(t.data) != first_len:
                raise ValueError("Size mismatch between tensors in TensorDataset at index " + str(idx))

        self.tensors = tensors
        self.length = first_len

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        item_list = []
        for t in self.tensors:
            item_list.append(t.data[index])
        return tuple(item_list)


def default_collate_fn(batch):
    # batch is a list of tuples: [(x1, y1), (x2, y2), ...]
    num_fields = len(batch[0])
    collated_outputs = []

    for field_idx in range(num_fields):
        field_samples = []
        for sample in batch:
            field_samples.append(sample[field_idx])

        # Pack into Tensor without gradients for inputs/targets
        collated_outputs.append(Tensor(field_samples, requires_grad=False))

    if num_fields == 1:
        return collated_outputs[0]
    return tuple(collated_outputs)


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, collate_fn=None, drop_last=False):
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be an instance of slowtorch.data.Dataset")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer, got " + str(batch_size))

        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.collate_fn = collate_fn if collate_fn is not None else default_collate_fn

    def __len__(self):
        total_items = len(self.dataset)
        if self.drop_last:
            return total_items // self.batch_size
        else:
            return (total_items + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))

        if self.shuffle:
            random.shuffle(indices)

        batch_buffer = []
        for idx in indices:
            batch_buffer.append(self.dataset[idx])
            if len(batch_buffer) == self.batch_size:
                yield self.collate_fn(batch_buffer)
                batch_buffer = []

        if len(batch_buffer) > 0 and not self.drop_last:
            yield self.collate_fn(batch_buffer)