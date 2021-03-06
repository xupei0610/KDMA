import os
import time
from abc import abstractmethod
import functools

import torch
from torch.utils.tensorboard import SummaryWriter

from . import utils

from typing import Iterable, Union, Sequence, Mapping, Optional
RealNum = Union[int, float]
from .agent import Agent, Hook
from .policy import Policy


class ActorCritic(Agent):

    def __init__(self, *,
            critic_network: torch.nn.Module,
            actor_network: Policy,
            header_network: torch.nn.Module = None,
            learning_rate: Optional[float] = None,
            critic_learning_rate: Optional[float] = None,
            actor_learning_rate: Optional[float] = None,
            header_learning_rate: Optional[float] = None,
            value_loss_coef: float = 1.,
            entropy_loss_coef: float = 0.,
            clip_grad_norm: Optional[float] = None,
            normalize_state: Optional[Sequence[int]] = None, # state shape
            clip_state: (Mapping[int, RealNum] or RealNum)=None,

            opt_epoch: int = 1,
            horizon: int = 1024,
            batch_size: int = 32,

            gamma: float = 0.99,

            checkpoint_save_interval: int = 1000,
            checkpoint_file: Optional[str] = None,
            log_dir: Optional[str] = None,
            max_iterations: Optional[int] = None,
            max_samples: Optional[int] = None,

            device: Optional[str] = None,
            is_chief: Optional[bool] = True,

            hooks: Iterable[Hook] = []
        ):
        super().__init__()
        self.opt_epoch = opt_epoch
        self.horizon = horizon
        self.batch_size = batch_size
        self.checkpoint_save_interval = checkpoint_save_interval
        self.checkpoint_file = checkpoint_file
        self.max_iterations = max_iterations
        self.max_samples = max_samples

        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_loss_coef = entropy_loss_coef
        self.clip_grad_norm = clip_grad_norm

        self.is_chief = is_chief
        for h in hooks:
            self.hooks.append(h)

        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        
        params = []
        self.main_params = []
        if critic_network:
            if not critic_learning_rate: critic_learning_rate = learning_rate
            params.append({"params": critic_network.parameters(), "lr": critic_learning_rate})
            self.main_params += list(critic_network.parameters())
        if actor_network:
            if not actor_learning_rate: actor_learning_rate = learning_rate
            params.append({"params": actor_network.parameters(),  "lr": actor_learning_rate})
            self.main_params += list(actor_network.parameters())
        if header_network:
            if not header_learning_rate: header_learning_rate = learning_rate
            params.append({"params": header_network.parameters(), "lr": header_learning_rate})
            self.main_params += list(header_network.parameters())
        if params:
            self.optimizers["main"] = torch.optim.Adam(params)
        
        self.model = torch.nn.Module()
        self.model.actor = actor_network
        self.model.critic = critic_network
        self.model.header = header_network
        if normalize_state:
            self.model.state_normalizer = utils.Normalizer(normalize_state, clip_state)
            if header_network:
                header_network.forward = functools.partial(utils.chain_fn, fns=(self.model.state_normalizer.__call__, header_network.forward))
            else:
                self.model.header = self.model.state_normalizer
            self.optimizers["state_normalizer"] = utils.MovingAverage(self.model.state_normalizer.parameters())
        else:
            self.model.state_normalizer = None

        if self.model.actor:
            self.actor_forward = self.model.actor.forward
            self.model.actor.forward = self.forward_actor
        if self.model.critic:
            self.critic_forward = self.model.critic.forward
            self.model.critic.forward = self.forward_critic

        self.optimizers["sample_counter"] = utils.Counter([self.samples])

        self.last_notify_time = 0.
        self.log_interval_time = 10.
        self._logger = log_dir
        
    def init(self):
        if self.model.state_normalizer is not None:
            self.hooks.insert(0, utils.MovingAverageNormalizerHook(
                self.model.state_normalizer, self.optimizers["state_normalizer"],
                lambda : self.buffer["state"]))

        if self.checkpoint_file:
            checkpoint_dir = os.path.dirname(self.checkpoint_file)
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # prevent multiprocessing race
            with utils.lock_file(self.checkpoint_file + ".lock"):
                if os.path.exists(self.checkpoint_file):
                    self.load_state_dict(torch.load(self.checkpoint_file, map_location=self.device))
                elif self.is_chief:
                    torch.save(self.state_dict(), self.checkpoint_file)
        
        self.to(self.device)
        self.global_step.data = self.global_step.to("cpu")
        self.samples.data = self.samples.to("cpu")

        for h in self.hooks: h.after_init(self)
        self.samples.grad = torch.zeros_like(self.samples)
        self._last_checkpoint_save_step = self.global_step.item()
        self._last_buffer_size = len(self.buffer)

        self.requests_quit = self.is_job_done()


    def forward(self, s, skip_header=False):
        if not skip_header:
            s = self.placeholder(s)
            if self.model.header:
                s = self.model.header(s)
        pi = self.model.actor(s, skip_header=True)
        v = self.model.critic(s, skip_header=True)
        if v.ndim > 1: v.squeeze_(1)
        return pi, v
    
    @property
    def actor(self):
        return self.model.actor

    def forward_actor(self, s, skip_header=False):
        if not skip_header:
            s = self.placeholder(s)
            if self.model.header:
                s = self.model.header(s)
        return self.actor_forward(s)
    
    @property
    def critic(self):
        return self.model.critic

    def forward_critic(self, s, skip_header=False):
        if not skip_header:
            s = self.placeholder(s)
            if self.model.header:
                s = self.model.header(s)
        return self.critic_forward(s)

    def clip_grad(self):
        if self.clip_grad_norm and self.main_params:
            torch.nn.utils.clip_grad_norm_(self.main_params, self.clip_grad_norm)
        
    def update(self):
        if self.optimizers["sample_counter"]:
            self.samples.grad.add_(len(self.buffer)-self._last_buffer_size)
        for h in self.hooks: h.before_update(self)
        vf_loss, pg_loss, entropy = self._update()
        if self.optimizers["sample_counter"]:
            self.step_opt(self.optimizers["sample_counter"])
            self.optimizers["sample_counter"].zero_grad()
        for h in self.hooks: h.after_update(self)

        global_step = self.global_step.item()
        self._last_buffer_size = len(self.buffer)
        if self.is_chief and self.checkpoint_file and global_step-self._last_checkpoint_save_step >= self.checkpoint_save_interval:
            torch.save(self.state_dict(), self.checkpoint_file)
            self._last_checkpoint_save_step = global_step

        if self.logger:
            if vf_loss is not None: self.logger.add_scalar("loss/vf_loss", vf_loss, global_step)
            if pg_loss is not None: self.logger.add_scalar("loss/pg_loss", pg_loss, global_step)
            if entropy is not None: self.logger.add_scalar("loss/entropy", entropy, global_step)
        # if self.is_chief:
        #     now = time.time()
        #     if now > self.log_interval_time + self.last_notify_time:
        #         print("[TRAIN] Steps: {:.0f}; Value Loss: {}; Policy Loss: {}; Entropy: {}; Samples: {:.0f}, {}".format(
        #             global_step,
        #             "None" if vf_loss is None else "{:.4f}".format(vf_loss), 
        #             "None" if pg_loss is None else "{:.4f}".format(pg_loss),
        #             "None" if entropy is None else "{:.4f}".format(entropy),
        #             self.samples.item(), time.strftime("%m-%d %H:%M:%S")
        #         ))
        #         self.last_notify_time = now
        if not self.requests_quit and self.is_job_done():
            self.requests_quit = True

    def _update(self):
        vf_loss, pg_loss, entropy = [], [], []
        for _ in range(self.opt_epoch):
            for batch in self.buffer:
                vf_loss_, pg_loss_, entropy_ = self.optimize(batch)
                if vf_loss_ is not None: vf_loss.append(vf_loss_)
                if pg_loss_ is not None: pg_loss.append(pg_loss_)
                if entropy_ is not None: entropy.append(entropy_)
                self.global_step += 1
        vf_loss = sum(vf_loss)/len(vf_loss) if vf_loss else None
        pg_loss = sum(pg_loss)/len(pg_loss) if pg_loss else None
        entropy = sum(entropy)/len(entropy) if entropy else None
        return vf_loss, pg_loss, entropy

    def optimize(self, data: Mapping[str, Sequence]) -> (float, float):
        losses = self.loss(data)
        vf_loss, pg_loss, entropy = losses[0], losses[1], losses[2]
        if vf_loss is None:
            loss = pg_loss
        elif pg_loss is None:
            loss = self.value_loss_coef*vf_loss
        else:
            loss = self.value_loss_coef*vf_loss + pg_loss
        if self.entropy_loss_coef and entropy:
            loss = loss - self.entropy_loss_coef*entropy
        if len(losses) > 3:
            loss = loss + sum(losses[3:])
        
        if self.optimizers["main"]:
            self.optimizers["main"].zero_grad()
        loss.backward()
        if self.optimizers["main"]:
            self.step_opt(self.optimizers["main"], before_step=self.clip_grad)
        return vf_loss.item() if torch.is_tensor(vf_loss) else vf_loss, \
            pg_loss.item() if torch.is_tensor(pg_loss) else pg_loss, \
            entropy.item() if torch.is_tensor(entropy) else entropy

    def state_dict(self) -> dict:
        state_dict = {
            "model": {k:v.cpu() for k, v in super().state_dict().items()},
        }
        for name, opt in self.optimizers.items():
            if opt:
                state_dict[name] = opt.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict: dict, strict=True):
        super().load_state_dict(state_dict["model"], strict)
        for name, opt in self.optimizers.items():
            if opt:
                opt.load_state_dict(state_dict[name])
    
    def to(self, device: (str or torch.device)):
        super().to(device)
        for opt in self.optimizers.values():
            if not opt or not hasattr(opt, "state"): continue
            for state in opt.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v) and v.device != device:
                        state[k] = v.to(device)
        self.device = device
        
    @property
    def logger(self):
        if self._logger and not isinstance(self._logger, SummaryWriter):
            self._logger = SummaryWriter(self._logger)
        return self._logger

    def is_job_done(self) -> bool:
        return (self.max_samples is not None and self.samples.item() >= self.max_samples) \
            or (self.max_iterations is not None and self.global_step.item() >= self.max_iterations)
    
    @property
    def requests_quit(self):
        return self._requests_quit

    @requests_quit.setter
    def requests_quit(self, v: bool):
        self._requests_quit = v
        if v and self.checkpoint_file and self.is_chief:
            torch.save(self.state_dict(), self.checkpoint_file) 