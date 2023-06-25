import numpy
import torch
import random
import os, time

from typing import Sequence


class FileLock(object):
    def __init__(self, lock_file, wait_time=1, timeout=3600):
        self.lock_file = lock_file
        self.wait_time = wait_time
        self.timeout = timeout
    # modified from
    # https://stackoverflow.com/questions/186202/what-is-the-best-way-to-open-a-file-for-exclusive-access-in-python
    def acquire(self):
        self._tmp_file = self.lock_file+"_tmp{}".format(os.getpid())
        open(self._tmp_file, "w+").close()
        start_time = time.time()
        elapsed_time = time.time() - start_time
        acquired = False
        while elapsed_time < self.timeout:
            if not os.path.exists(self.lock_file):
                try:
                    if os.name != 'nt': # non-windows needs a create-exclusive operation
                        fd = os.open(self.lock_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
                        os.close(fd)
                    os.rename(self._tmp_file, self.lock_file)
                    acquired = True
                except (OSError, ValueError, IOError) as e:
                    if os.name != 'nt' and not 'File exists' in str(e): raise
            if acquired:
                break
            elapsed_time = time.time() - start_time
            time.sleep(self.wait_time)
    def release(self):
        os.remove(self.lock_file)
        if os.path.exists(self._tmp_file):
            os.remove(self._tmp_file)
    def __enter__(self):
        self.acquire()
    def __exit__(self, type, value, traceback):
        self.release()

def lock_file(lock_file: str, wait_time=1, timeout=3600):
    return FileLock(lock_file, wait_time, timeout)


def env_overtime(info: dict):
    return "TimeLimit.truncated" in info and info["TimeLimit.truncated"]

def env_seed(env, seed: int):
    if env and hasattr(env, "seed"):
        env.seed(seed)
    else:
        print("[Warn] No `seed` method find when seeding environment.")

def seed(seed: int, env=None):
    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder="big")
        numpy.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        env_seed(env, seed)
        return
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.set_deterministic(True)
    except AttributeError:
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass
    if env is not None:
        env_seed(env, seed)


from numba import jit
@jit(nopython=True)
def discount(arr: Sequence[float], discount_factor: float):
    result = [0.]*len(arr)
    result[-1] = arr[-1]
    for i in range(len(arr)-2, -1, -1):
        result[i] = arr[i] + discount_factor*result[i+1]
    return result


class MovingAverage(torch.optim.Optimizer):
    def __init__(self, params, decay=0.9999):
        defaults=dict(decay=decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            decay = group["decay"]
            for p in group["params"]:
                p.mul_(decay).add_(p.grad, alpha=1-decay)
        return loss

class Counter(torch.optim.Optimizer):
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
                p.add_(p.grad)
        return loss

from .agent import Hook
class MovingAverageNormalizerHook(Hook):
    def __init__(self, normalizer, optimizer, data):
        self.normalizer = normalizer
        self.optimizer = optimizer
        self.data = data
    def before_update(self, model):
        if self.optimizer:
            s = model.placeholder(self.data() if callable(self.data) else self.data, device=self.normalizer.mean.device)
            s = s.view(-1, *self.normalizer.mean.shape)
            if self.normalizer.mean.grad is None:
                self.normalizer.mean.grad = s.mean(0).detach()
                self.normalizer.std.grad = s.std(0).detach()
            else:
                self.normalizer.mean.grad.data.copy_(s.mean(0))
                self.normalizer.std.grad.data.copy_(s.std(0))
            # step = self.normalizer.counter.item()
            step = model.global_step.item()/model.opt_epoch
            self.optimizer.defaults["decay"] = min(0.9999, (step+1)/(step+10))
            for group in self.optimizer.param_groups:
                group["decay"] = self.optimizer.defaults["decay"]
            model.step_opt(self.optimizer)
            # self.normalizer.counter += 1

class Normalizer(torch.nn.Module):
    def __init__(self, shape, clamp=None):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        self.std = torch.nn.Parameter(torch.ones(shape, dtype=torch.float32))
        # self.register_buffer("counter", torch.as_tensor(0., dtype=torch.float32))
        if type(clamp) == float:
            self.clamp = (-abs(clamp), abs(clamp)) if clamp else None
        else:
            self.clamp = clamp
    def forward(self, s):
        with torch.no_grad():
            s = (s-self.mean)/self.std.clamp(min=1e-8)
            if self.clamp: s.clamp_(self.clamp[0], self.clamp[1])
        return s
    
def chain_fn(x, fns):
    for fn in fns: x = fn(x)
    return x
