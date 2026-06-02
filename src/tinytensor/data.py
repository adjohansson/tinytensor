import numpy as np
import numpy.typing as npt

from typing import Iterator


class ArrayDataset:
    def __init__(self, *arrays: npt.NDArray[np.float32]):
        self._arrays = arrays
        self._len = arrays[0].shape[0]
        for array in arrays:
            if not array.shape[0] == self._len:
                raise ValueError("Tried to create ArrayDataset with inconsistend number of datapoints.")

    def __getitem__(self, idx: npt.NDArray[np.long]) -> tuple[npt.NDArray[np.float32], ...]:
        return tuple(array[idx, ...] for array in self._arrays)

    def __len__(self) -> int:
        return self._len


class DataLoader:
    def __init__(self, dataset: ArrayDataset, batch_size: int, shuffle: bool):
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[tuple[npt.NDArray, ...]]:
        if self.shuffle:
            permutation = np.random.permutation(len(self.dataset))
        else:
            permutation = np.arange(len(self.dataset))

        for i in range(len(self.dataset) // self.batch_size):
            idx = slice(i*self.batch_size, (i+1)*self.batch_size)
            yield self.dataset[permutation[idx]]

    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size


def onehot(y: npt.NDArray[np.long], n_classes: int) -> npt.NDArray:
    assert y.ndim == 1
    len_ = y.shape[0]
    a = np.zeros((len_, n_classes), dtype=np.float32)
    a[np.arange(len_), y] = 1.0
    return a

