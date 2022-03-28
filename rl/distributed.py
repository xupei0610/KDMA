import os
import torch

from typing import Callable
from .agent import Hook


def worker_name(rank: int):
    name = "Evaluator" if rank < 0 else "Worker"
    return "{}{}".format(name, abs(rank))

class DistributedSyncHook(Hook):
    # sync parameter from chief worker
    def after_init(self, model):
        rank = torch.distributed.get_rank()
        chief_rank = rank if model.is_chief else 0
        chief_rank = torch.as_tensor(chief_rank)
        torch.distributed.all_reduce(chief_rank.data, op=torch.distributed.ReduceOp.SUM, async_op=False)
        chief_rank = chief_rank.item()
        handles = []
        for p in model.parameters():
            handles.append(
                torch.distributed.broadcast(p.data, chief_rank, async_op=True)
            )
        for h in handles: h.wait()

    # average gradient from all workers
    def before_opt(self, model, opt):
        world_size = torch.distributed.get_world_size()
        handles, updated_params = [], []
        for group in opt.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    handles.append(
                        torch.distributed.all_reduce(p.grad.data, op=torch.distributed.ReduceOp.SUM, async_op=True)
                    )
                    updated_params.append(p)
        reduce_mean = []
        for p in updated_params:
            if not any(p is p_ for p_ in model.shared_parameters()):
                reduce_mean.append(p)
        for h in handles:
            h.wait()
        for p in reduce_mean:
            p.grad.div_(world_size)

def distributed(fn: Callable,
    backend: torch.distributed.Backend,
    rank: int,
    world_size: int,
    fn_kwargs=dict(), 
    master_addr: str = "127.0.0.1",
    master_port: (str or int) = "29501"
):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    name = worker_name(rank)
    torch.distributed.init_process_group(backend, rank=rank, world_size=world_size)
    fn(**fn_kwargs)

