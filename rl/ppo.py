import torch
import numpy
import time

from . import utils
from .agent import Hook
from .actor_critic import ActorCritic
from .buffer import OnPolicyBuffer

from typing import Sequence, Optional


class Synchronizer(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                p.copy_(p.grad)
        return loss

class PPO(ActorCritic):

    def __init__(self, *, lambd: float = 0.95, clip_range: float = 0.2, value_clip_range: float = None, **kwargs):
        super().__init__(**kwargs)

        self.gamma_lambda = self.gamma*lambd
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range

        self.buffer = OnPolicyBuffer([
            "state", "action", "value", "advantage", "log_prob"
        ], capacity=self.horizon, batch_size=self.batch_size)
        self._buffer_path_ptr = 0

        self.advantage_normalizer = torch.nn.Module()
        self.advantage_normalizer.mean = torch.nn.Parameter(torch.tensor(0., dtype=torch.float32, device=self.device))
        self.advantage_normalizer.std = torch.nn.Parameter(torch.tensor(1., dtype=torch.float32, device=self.device))
        self.optimizers["advantage_normalizer_mean"] = Synchronizer([self.advantage_normalizer.mean])
        self.optimizers["advantage_normalizer_std"] = Synchronizer([self.advantage_normalizer.std])
        
    def act(self, s, stochastic: Optional[bool]=None) -> Sequence:
        with torch.no_grad():
            pi, v = self(s)
            a = pi.sample() if stochastic or (stochastic is None and self.training) else pi.mean
            lp = pi.log_prob(a).sum()
        return a.cpu().detach_().squeeze_(0).numpy(), lp.item(), v.item()

    def store(self, s, a, r, s_, done, info, log_prob, v):
        buffer_size = len(self.buffer)
        if self._buffer_path_ptr > buffer_size: self._buffer_path_ptr = buffer_size
        self.buffer.store(
            state=s, action=a, value=v, advantage=r, log_prob=log_prob
        )
        self._needs_update = len(self.buffer) >= self.horizon if self.horizon else done
        if done or self._needs_update:
            self.update_path_advantage(s_ if utils.env_overtime(info) or not done else None)

    def needs_update(self) -> bool:
        return self._needs_update

    def _update(self):
        adv = self.placeholder(self.buffer["advantage"])
        if self.advantage_normalizer.mean.grad is None:
            self.advantage_normalizer.mean.grad = adv.mean().detach()
        else:
            self.advantage_normalizer.mean.grad.data.copy_(adv.mean())
        self.step_opt(self.optimizers["advantage_normalizer_mean"])
        std = (adv-self.advantage_normalizer.mean).square().mean().sqrt()
        if self.advantage_normalizer.std.grad is None:
            self.advantage_normalizer.std.grad = std.detach()
        else:
            self.advantage_normalizer.std.grad.data.copy_(std)
        self.step_opt(self.optimizers["advantage_normalizer_std"])
        
        res = super()._update()
        self.buffer.clear()
        return res

    def loss(self, data) -> (torch.Tensor, torch.Tensor):
        pi_, v_ = self(data["state"])
        a = self.placeholder(data["action"])
        v = self.placeholder(data["value"])
        lp = self.placeholder(data["log_prob"])
        adv = self.placeholder(data["advantage"])
            
        with torch.no_grad():
            v_t = v + adv
            adv = (adv-self.advantage_normalizer.mean)/(self.advantage_normalizer.std + 1e-8)

        vf_loss = (v_ - v_t).square().mean()
        if self.value_clip_range:
            clipped_v_ = v + torch.clamp(v_ - v, -self.value_clip_range, self.value_clip_range)
            vf_loss2 = (clipped_v_ - v_t).square()
            vf_loss = torch.max(vf_loss, vf_loss2).mean()

        lp_ = pi_.log_prob(a).sum(1)
        ratio = (lp_ - lp).exp()
        clipped_ratio = ratio.clamp(1.-self.clip_range, 1.+self.clip_range)
        pg_loss = -torch.min(adv*ratio, adv*clipped_ratio).mean()

        entropy = pi_.entropy().sum(1).mean() if self.entropy_loss_coef else None
        return vf_loss, pg_loss, entropy

    def update_path_advantage(self, bootstrap_state, end_index=None):
        if bootstrap_state is None:
            v_ = 0.
        else:
            v_ = self.critic(bootstrap_state).item()
        value = numpy.append(self.buffer["value"][self._buffer_path_ptr:end_index], v_)
        reward = self.buffer["advantage"][self._buffer_path_ptr:end_index]
        td_err = reward + self.gamma*value[1:] - value[:-1]
        self.buffer["advantage"][self._buffer_path_ptr:end_index] = numpy.float32(utils.discount(td_err, self.gamma_lambda))
        self._buffer_path_ptr += len(reward)


