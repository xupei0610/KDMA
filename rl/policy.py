import torch
import numpy
from typing import Sequence, Optional
from abc import ABCMeta, abstractmethod

class Distribution(ABCMeta):
    @abstractmethod
    def to(self, device: str or torch.device):
        raise NotImplementedError

class NormalDistribution(torch.distributions.Normal, metaclass=Distribution):
    def to(self, device: str or torch.device):
        self.loc.to(device)
        self.scale.to(device)
        return self

    def is_cuda(self) -> bool:
        return self.loc.is_cuda



class Policy(torch.nn.Module):

    def forward(self, s) -> torch.distributions.distribution:
        raise NotImplementedError

class GaussianPolicy(Policy):

    def __init__(self,
        in_features: int, action_dim: int,
        *, log_std: Optional[torch.Tensor] = None,
        init_std: Optional[(float or Sequence[float])] = None
    ):
        super().__init__()
        self.policy_mu = torch.nn.Linear(in_features, action_dim)
        if log_std is None:
            self.policy_log_std = torch.nn.Linear(in_features, action_dim)
        else:
            self.policy_log_std = log_std
        if hasattr(float, "__iter__"):
            init_std = numpy.asarray(init_std, dtype=numpy.float32)
            assert(all(std > 0 for std in init_std))
        else:        
            assert(init_std is None or init_std > 0)
        if init_std is not None:
            init_log_std = numpy.log(init_std)
            if torch.is_tensor(self.policy_log_std):
                torch.nn.init.constant_(self.policy_log_std, init_log_std)
            else:
                torch.nn.init.constant_(self.policy_log_std.weight, 0.)
                torch.nn.init.constant_(self.policy_log_std.bias, init_log_std)

    def forward(self, x) -> torch.distributions.Normal:
        loc = self.policy_mu(x)
        if torch.is_tensor(self.policy_log_std):
            std = torch.exp(self.policy_log_std)
        else:
            std = torch.exp(self.policy_log_std(x))
        try:
            return NormalDistribution(loc, std, validate_args=True)
        except Exception as e:
            print("Invalid Value", x, loc, std)
            raise e
