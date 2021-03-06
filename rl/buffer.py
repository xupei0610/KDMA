import torch
from abc import ABCMeta, abstractmethod
from typing import Iterable, Sequence, Mapping, Optional

import numpy
class Buffer(metaclass=ABCMeta):

    def __init__(self,
        keys: Iterable[str],
        *,
        capacity: Optional[int] = None,
        batch_size: int = None,
        shuffle: bool = True
    ):
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._capacity = capacity
        self._exp = {
            k: [None]*(capacity if capacity else 4096) for k in keys
        }
        self.clear()

    @property
    def batch_size(self) -> int:
        if self._batch_size and self._batch_size > 0:
            return self._batch_size
        else:
            return self.__len__() 
    
    @batch_size.setter
    def batch_size(self, value: int):
        self._batch_size = value
    
    @property
    def shuffle(self) -> bool:
        return self._shuffle
    
    @shuffle.setter
    def shuffle(self, value: bool):
        self._shuffle = value

    def add(self, key: str):
        if key not in self._exp.keys():
            self._exp[key] = [None]*(self._capacity if self._capacity else 4096)
            self._exp[key].clear()

    def clear(self):
        for e in self._exp.values():
            e.clear()
    
    def keys(self) -> Iterable[str]:
        return self._exp.keys()
    
    def values(self) -> Iterable[Sequence]:
        return self._exp.values()
    
    def items(self) -> Mapping[str, Sequence]:
        return self._exp.items()

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def __iter__(self):
        raise NotImplementedError


class OnPolicyBuffer(Buffer, torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        Buffer.__init__(self,  *args, **kwargs)
        self._len = 0
        self._iterator = None
        self.collate_fn = None

    def __len__(self) -> int:
        return self._len

    def store(self, **kwargs):
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                self._exp[k].append(v)
            elif hasattr(v, "__len__"):
                self._exp[k].append(numpy.asarray(v, dtype=numpy.float32))
            elif type(v) == float:
                self._exp[k].append(numpy.float32(v))
            else:
                self._exp[k].append(v)
        self._len = max(len(_) for _ in self._exp.values())

    def __getitem__(self, idx: (str or int)) -> (Sequence or dict):
        if type(idx) == str:
            return self._exp[idx]
        return {
            k: v[idx] for k, v in self._exp.items()
        }

    def clear(self):
        super().clear()
        self._len = 0

    def __iter__(self):
        if not self._iterator:
            self._iterator = torch.utils.data.DataLoader(self, shuffle=self.shuffle, batch_size=self.batch_size, collate_fn=self.collate_fn)
        return self._iterator.__iter__()

    @Buffer.batch_size.setter
    def batch_size(self, value: int):
        if self._batch_size != value and self._iterator is not None:
            self._iterator = None
        self._batch_size = value

    @Buffer.shuffle.setter
    def shuffle(self, value: bool):
        if self._shuffle != value and self._iterator is not None:
            self._iterator = None
        self._shuffle = value
