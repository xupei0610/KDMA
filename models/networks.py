from typing import Sequence
import torch
import numpy

class Policy(torch.nn.Module):
    def __init__(self, agent_dim: int, neighbor_dim: int, out_dim: int):
        super().__init__()
        self.agent_dim = agent_dim
        self.neighbor_dim = neighbor_dim
        self.neighbor_embedding = torch.nn.Sequential(
            torch.nn.Linear(self.neighbor_dim, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256)
        )
        self.agent_embedding = torch.nn.Sequential(
            torch.nn.Linear(self.agent_dim, 256),
        )
        self.model = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, out_dim)
        )
    
    def embedding(self,
        agent: torch.Tensor,
        neighbors: Sequence[torch.Tensor] or torch.Tensor,
        n_neighbors: Sequence[int] or torch.Tensor = None
    ) -> torch.Tensor:
        if agent.ndim > 1:
            if torch.is_tensor(neighbors):
                x = self.neighbor_embedding(neighbors)
                if n_neighbors is not None:
                    seq_mask = torch.arange(x.size(1), device=n_neighbors.device) < n_neighbors.view(-1, 1)
                    x = x*seq_mask.unsqueeze(-1)
                x = x.sum(1)
            else:
                x = self.neighbor_embedding(torch.cat([_ for _ in neighbors], axis=0))
                x = torch.split(x, [_.size(0) for _ in neighbors])
                if n_neighbors is None:
                    x = torch.stack([_.sum(0) for _ in x])
                else:
                    x = torch.stack([_[:n].sum(0) for _, n in zip(x, neighbors)])
            x = x + self.agent_embedding(agent)
        else:
            assert(torch.is_tensor(neighbors))
            x = self.neighbor_embedding(neighbors.view(-1, self.neighbor_dim)).sum(0)
            x = x + self.agent_embedding(agent)
        return x

    def forward(self,
        agent: torch.Tensor,
        neighbors: Sequence[torch.Tensor] or torch.Tensor,
        n_neighbors: Sequence[int]=None
    ) -> torch.Tensor:
        y = self.model(self.embedding(agent, neighbors, n_neighbors))
        return y

    def placeholder(self, arr, device=None, dtype=torch.float32) -> torch.Tensor:
        with torch.no_grad():
            if not torch.is_tensor(arr):
                if hasattr(arr, "__len__") and torch.is_tensor(arr[0]):
                    arr = torch.cat(arr)
                else:
                    arr = torch.as_tensor(arr, dtype=dtype)
            return arr.to(next(self.parameters()).device if device is None else device)
    
ExpertNetwork = Policy

class ActorNetwork(torch.nn.Module):
    def __init__(self, init_std:float=.5):
        super().__init__()
        self.n_neighbors = None # sequence length for batch update
        self.model = Policy(agent_dim=3, neighbor_dim=4, out_dim=2)
        self.log_std = torch.nn.Parameter(torch.tensor([
            numpy.log(init_std), numpy.log(init_std)
        ], dtype=torch.float32))

        self.placeholder = self.model.placeholder

    def forward(self, s, neighbors=None, n_neighbors=None):
        if neighbors is None:
            if s.ndim == 1:
                agent, neighbors = s[:3], s[3:].view(-1, 4)
            else: #s.ndim == 2
                agent, neighbors = s[:, :3], s[:, 3:].view(s.size(0), -1, 4)
            mu = self.model(agent, neighbors, self.n_neighbors)
        else:
            mu = self.model(s, neighbors, self.n_neighbors if n_neighbors is None else n_neighbors)
        std = self.log_std.exp()
        return torch.distributions.Normal(mu, std)


class CriticNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_neighbors = None # sequence length for batch update
        self.model = Policy(agent_dim=3, neighbor_dim=4, out_dim=1)
        
    def forward(self, s):
        if s.ndim == 1:
            agent, neighbors = s[:3], s[3:].view(-1, 4)
        else: #s.ndim == 2
            agent, neighbors = s[:, :3], s[:, 3:].view(s.size(0), -1, 4)
        return self.model(agent, neighbors, self.n_neighbors)
