from typing import Sequence, Iterable, Optional
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from .buffer import Buffer
import torch

class Agent(torch.nn.Module, metaclass=ABCMeta):

    def __init__(self):
        self.device = None
        self.requests_quit = False
        self.hooks = []
        self.optimizers = OrderedDict()
        self.buffer: Buffer = None
        self.model: torch.nn.Module = None
        self._shared_parameters = []

        super().__init__()
        self.register_buffer("global_step", torch.tensor(0, dtype=torch.float64, device="cpu"))
        self.samples = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64, device="cpu"))
        self._shared_parameters.append(self.samples)

    def shared_parameters(self) -> Iterable[torch.Tensor]:
        return iter(self._shared_parameters)

    def placeholder(self, arr, device=None, dtype=torch.float32) -> torch.Tensor:
        with torch.no_grad():
            if not torch.is_tensor(arr):
                if hasattr(arr, "__len__") and torch.is_tensor(arr[0]):
                    arr = torch.cat(arr)
                else:
                    arr = torch.as_tensor(arr, dtype=dtype)
            return arr.to(self.device if device is None else device)

    def step_opt(self, opt, before_step=None, after_step=None):
        if opt:
            no_opt = False
            for h in self.hooks: no_opt |= h.before_opt(self, opt) is False
            if not no_opt:
                if before_step is not None: before_step()
                opt.step()
                if after_step is not None: after_step()
            for h in self.hooks: h.after_opt(self, opt)

    def init(self):
        pass

    @abstractmethod
    def act(self, s, stochastic: bool=None) -> (Sequence, Sequence):
        pass

    @abstractmethod
    def store(self, s, a, r, s_, done, info, *args):
        pass

    @abstractmethod
    def loss(self, data) -> (torch.Tensor, torch.Tensor):
        pass

    @abstractmethod
    def needs_update(self) -> bool:
        pass 

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def optimize(self, data):
        pass


class Hook:
    def after_init(self, model: Agent):
        pass
    def before_update(self, model: Agent):
        pass
    def after_update(self, model: Agent):
        pass
    def before_opt(self, model:Agent, opt: torch.optim.Optimizer) -> Optional[bool]:
        pass
    def after_opt(self, model:Agent, opt: torch.optim.Optimizer):
        pass
