from functools import partial
from typing import Sequence
import os, time, multiprocessing
import torch

from rl import utils
from models.ppo import MultiAgentPPO
from models.networks import ExpertNetwork

from env.scenarios import *
from models.agent import DLAgent
from models.env import Env

import config

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--expert", type=str, default=None)
parser.add_argument("--log_dir", type=str, default=None)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--rank", type=int, default=None)
parser.add_argument("--master_addr", type=str, default="127.0.0.1")
parser.add_argument("--master_port", type=str, default="29501")
parser.add_argument("--workers", type=int, default=1)
settings = parser.parse_args()


def env_wrapper(expert=None, evaluate=False):
    agent_wrapper = partial(DLAgent,
        preferred_speed=config.PREFERRED_SPEED, max_speed=config.MAX_SPEED,
        radius=config.AGENT_RADIUS, observe_radius=config.NEIGHBORHOOD_RADIUS,
        expert=expert
        )
    if evaluate:
        scenario = CircleCrossingScenario(n_agents=20, agent_wrapper=agent_wrapper, min_distance=0.3, radius=4)
    else:
        kwargs = dict(n_agents=(6, 20), agent_wrapper=agent_wrapper, min_distance=0.3)
        scenario = CompositeScenarios([
            CircleCrossingScenario(radius=(4, 6), noise=0.5, **kwargs),
            SquareCrossingScenario(width=(8, 12), height=(8, 12), vertical=True, horizontal=True, **kwargs),
            SquareCrossingScenario(width=(8, 12), height=(8, 12), vertical=True, horizontal=False, **kwargs),
            SquareCrossingScenario(width=(8, 12), height=(8, 12), vertical=False, horizontal=True, **kwargs)
        ])
    env = Env(scenario=scenario, fps=1/config.STEP_TIME, timeout=config.TIMEOUT, frame_skip=config.FRAME_SKIP)
    return env


def agent_wrapper(env, rank, hooks=[]):
    is_chief = rank == 0
    is_evaluator = rank < 0
    ckpt = os.path.join(settings.log_dir, "ckpt") if settings.log_dir else None
    log_dir = settings.log_dir if is_chief or is_evaluator else None

    agent = MultiAgentPPO(
        actor_learning_rate=config.ACTOR_LR,
        critic_learning_rate=config.CRITIC_LR,
        entropy_loss_coef=config.ENTROPY_LOSS_COEF,
        clip_grad_norm=config.GRAD_NORM_CLIP,
        gamma=config.DISCOUNT_FACTOR,
        opt_epoch=config.OPT_EPOCHS,
        horizon=config.HORIZON,
        batch_size=config.BATCH_SIZE,
        max_samples=config.MAX_SAMPLES,
        init_action_std=config.INIT_ACTION_STD,

        checkpoint_file=ckpt,
        log_dir=log_dir,
        device=settings.device,
        is_chief=is_chief,
        hooks=[h() for h in hooks]
    )
    agent.rank = rank
    return agent


def train(seed, rank, hooks=[]):
    utils.seed(seed)
    agent = agent_wrapper(None, rank, hooks)
    agent.init()

    expert = ExpertNetwork(agent_dim=4, neighbor_dim=4, out_dim=2)
    if os.path.isdir(settings.expert):
        expert_ckpt = os.path.join(settings.expert, "ckpt")
    else:
        expert_ckpt = settings.expert
    ckpt = torch.load(expert_ckpt, map_location="cpu")
    expert.load_state_dict(ckpt["model"])
    expert.to(agent.device)
    
    env = env_wrapper(expert)
    env.seed(seed)

    agent.eval()
    done = True
    if settings.workers > 0:
        print("Worker {} starts work.".format(rank))
    while not agent.requests_quit:
        if done:
            s = env.reset()
        else:
            s = s_
        a, *args = agent.act(s, True)
        
        act = [ag.act(ac, env) for ag, ac in zip(env.agents, a)]
        s_, r, done, info = env.step(act)
        agent.store(s, a, r, s_, done, info, *args)

        if agent.needs_update():
            agent.train()
            agent.update()
            agent.eval()

def evaluate(
    seed: int = None,
    trials: int = 10,
    child_processes: Sequence[multiprocessing.context.Process] = None,
    timeout: int = 3600,
    keep_best_checkpoint: bool = True
):

    env = env_wrapper(evaluate=True)
    agent = agent_wrapper(env, rank=-1)
    agent.init()
    agent.eval()

    done = True
    tries = -1
    best_reward = -999999
    finished, global_step = False, -1

    total, lifetime, arrived, collided = 0, 0, 0, 0
    avg_reward, speed = [], []
    while True:
        if tries < 0:
            last_response_time = time.time()
            ckpt = None
            while not finished:
                if os.path.exists(agent.checkpoint_file):
                    try:
                        ckpt = torch.load(agent.checkpoint_file)
                    except:
                        pass
                    if ckpt:
                        agent.load_state_dict(ckpt)
                        step = agent.global_step.item()
                        if step <= global_step:
                            ckpt = None
                        else:
                            global_step = step
                            break
                finished = False
                if child_processes:
                    finished = not all(p.is_alive() for p in child_processes)
                if not finished and timeout and timeout > 0:
                    finished = time.time() - last_response_time > timeout
                if not finished: time.sleep(30)
            if finished: break
            tries = 0
            utils.env_seed(env, seed)

        if done:
            s = env.reset()
            reward = [[] for _ in range(len(env.agents))]
        else:
            s = s_
        a = agent.act(s, False)[0]

        act = [ag.act(ac, env) for ag, ac in zip(env.agents, a)]
        s_, rews, done, info = env.step(act)

        for idx, (_, r, ag) in enumerate(zip(s, rews, env.agents)):
            if _ is not None:
                reward[idx].append(r)
                speed.append((ag.velocity.x**2 + ag.velocity.y**2)**0.5)
                lifetime += 1
                
        tries += done
        if done:
            arrived += len(info["arrived_agents"])
            collided += len(info["collided_agents"])
            total += len(env.agents)
            rews = []
            for r in reward:
                rews.append(r[-1])
                for _ in reversed(r[:-1]):
                    rews.append(_ + agent.gamma*rews[-1])
            avg_reward.append(sum(rews)/len(rews))
        
        if tries >= trials:
            success_rate = arrived/total
            collision_rate = collided/total
            avg_time = lifetime/total
            avg_speed = sum(speed)/len(speed)
            avg_reward = sum(avg_reward)/len(avg_reward)
            samples = agent.samples.item()
            if agent.logger:
                agent.logger.add_scalar("eval/success_rate", success_rate, global_step)
                agent.logger.add_scalar("eval/collision_rate", collision_rate, global_step)
                agent.logger.add_scalar("eval/avg_time", avg_time, global_step)
                agent.logger.add_scalar("eval/avg_speed", avg_speed, global_step)
                agent.logger.add_scalar("eval/avg_reward", avg_reward, global_step)
                agent.logger.add_scalar("model/samples", agent.samples.item(), global_step)
                std = agent.actor.log_std.exp().cpu().tolist()
                agent.logger.add_scalar("model/std_x", std[0], global_step)
                agent.logger.add_scalar("model/std_y", std[1], global_step)
            print("[PERFORM] Step: {:.0f}, Collision: {:.2f}, Success: {:.2f}, Reward: {:.2f}, Samples: {:.0f}, {}".format(
                global_step, collision_rate, success_rate, avg_reward,
                samples, time.strftime("%m-%d %H:%M:%S")
            ))
            total, lifetime, arrived, collided = 0, 0, 0, 0
            avg_reward, speed = [], []
            tries = -1

            cache_id = int(samples) // int(5e6)
            if cache_id:
                cache_file = agent.checkpoint_file+"-{}".format(cache_id)
                if not os.path.exists(cache_file):
                    torch.save(ckpt, cache_file)

if __name__ == "__main__":
    if settings.workers > 1:
        assert(settings.workers > 1 and settings.rank is not None)
        from rl.distributed import distributed, DistributedSyncHook
        if settings.rank == 0:
            processes = []
            torch.multiprocessing.set_start_method("spawn", force=True)
            p = torch.multiprocessing.Process(target=distributed, args=(
                partial(train, seed=1, rank=0, hooks=[DistributedSyncHook]),
                "gloo", settings.rank, settings.workers
            ), kwargs=dict(master_addr=settings.master_addr, master_port=settings.master_port))
            p.start()
            processes.append(p)
            evaluate(seed=1+settings.workers, child_processes=processes)
        else:
            distributed(
                partial(train, seed=1+settings.rank, rank=settings.rank, hooks=[DistributedSyncHook]),
                "gloo", settings.rank, settings.workers,
                master_addr=settings.master_addr, master_port=settings.master_port
            )
    else:
        processes = []
        torch.multiprocessing.set_start_method("spawn", force=True)
        p = torch.multiprocessing.Process(target=train, kwargs=dict(seed=1, rank=0))
        p.start()
        processes.append(p)
        evaluate(seed=1+settings.workers, child_processes=processes)
