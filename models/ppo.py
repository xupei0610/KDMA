
from typing import Optional
import torch
import numpy

from rl.ppo import PPO
from .networks import CriticNetwork, ActorNetwork


class MultiAgentPPO(PPO):
    
    def __init__(self, init_action_std, *args, **kwargs):
        super().__init__(
            actor_network=ActorNetwork(init_action_std),
            critic_network=CriticNetwork(),
            *args, **kwargs
        )
        self.buffer.add("n_neighbors")
        self.buffer.collate_fn = self.state_zero_pad
        self.exp_cache = [] 
        self.agents_done = []
    
    @staticmethod
    def state_zero_pad(batch):
        max_len = 0
        for entry in batch:
            max_len = max(max_len, len(entry["state"]))
        for entry in batch:
            if max_len > len(entry["state"]):
                entry["state"] = numpy.pad(entry["state"], (0, max_len-len(entry["state"])))
        return {
            k: torch.tensor(numpy.array([e[k] for e in batch])) for k, v in entry.items()
        }


    def act(self, states, stochastic: Optional[bool]=None):
        act, vs, lps = [], [], []
        for s in states:
            if s is None:     # stopped agents
                a, v, lp = None, None, None
            else:
                with torch.no_grad():
                    pi, v = self(s)
                    a = pi.sample() if stochastic or (stochastic is None and self.training) else pi.mean
                    lp = pi.log_prob(a).sum()
            
                a = a.cpu().numpy()
                v, lp = v.item(), lp.item()
            act.append(a)
            vs.append(v)
            lps.append(lp)
        return act, lps, vs

    def store(self, states, acts, rews, states_, terminal, info, lps, vs):
        boostrasp_states = []

        for idx, (s, a, r, s_, lp, v) in enumerate(zip(states, acts, rews, states_, lps, vs)):
            if idx == len(self.exp_cache):
                self.exp_cache.append(dict(advantage=[], value=[], idx=[]))
            exp = self.exp_cache[idx]

            # agent already stopped
            if s is None: 
                # assert(not exp["idx"])
                boostrasp_states.append(None)
                continue
            
            boostrasp_states.append(s_)
            exp["advantage"].append(r)
            exp["value"].append(v)

            if not self.horizon or len(self.buffer) < self.horizon:
                # store experience sample if replay buffer is not full
                self.buffer.store(state=s, action=a, value=v, advantage=0., log_prob=lp, n_neighbors=(len(s)-self.critic.model.agent_dim)//self.critic.model.neighbor_dim)
                exp["idx"].append(len(self.buffer)-1)
            else:
                # replay buffer is full
                exp["idx"].append(None)
            
        self._needs_update = len(self.buffer) >= self.horizon if self.horizon else terminal

        for s_, exp in zip(boostrasp_states, self.exp_cache):
            if not exp["idx"]: continue
            if self._needs_update or terminal or s_ is None:
                # set the current agent's cached experience buffer as the default buffer
                exp, self.buffer = self.buffer, exp
                # compute advantage for the current agent
                self._buffer_path_ptr = 0
                self.update_path_advantage(s_)
                exp, self.buffer = self.buffer, exp
                # store advantage
                for i, adv in zip(exp["idx"], exp["advantage"]):
                    if i is None: break
                    self.buffer["advantage"][i] = adv
                exp["advantage"].clear()
                exp["value"].clear()
                exp["idx"].clear()


    def loss(self, data) -> (torch.Tensor, torch.Tensor):
        n_neighbors = self.placeholder(data["n_neighbors"])
        self.actor.n_neighbors = n_neighbors
        self.model.n_neighbors = n_neighbors
        res = super().loss(data)
        self.actor.n_neighbors = None
        self.model.n_neighbors = None
        return res

