from functools import partial
import os, time
import torch

from env.scenarios import *
from models.networks import ExpertNetwork
from models.env import Env
from models.agent import DLAgent

import config

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, default=None)
parser.add_argument("--max_trials", type=int, default=50)
parser.add_argument("--scene", type=str, default="6-circle",
    choices=["6-circle", "12-circle", "20-circle", "24-circle", "20-corridor", "24-corridor", "20-square", "24-square"]
)
parser.add_argument("--device", type=str, default=None)
settings = parser.parse_args()

def env_wrapper(model, expert=None):
    agent_wrapper = partial(DLAgent,
        preferred_speed=config.PREFERRED_SPEED, max_speed=config.MAX_SPEED,
        radius=config.AGENT_RADIUS, observe_radius=config.NEIGHBORHOOD_RADIUS,
        expert=expert, model=model
        )
    if settings.scene == "6-circle": 
        scenario = CircleCrossing6Scenario(agent_wrapper=agent_wrapper)
    elif settings.scene == "12-circle":
        scenario = CircleCrossing12Scenario(agent_wrapper=agent_wrapper)
    elif settings.scene == "20-circle":
        scenario = CircleCrossingScenario(n_agents=20, agent_wrapper=agent_wrapper, min_distance=0.3, radius=4)
    elif settings.scene == "24-circle":
        scenario = CircleCrossingScenario(n_agents=24, agent_wrapper=agent_wrapper, min_distance=0.3, radius=4)
    elif settings.scene == "20-square":
        scenario = SquareCrossingScenario(n_agents=20, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=True, horizontal=True, width=8, height=8)
    elif settings.scene == "24-square":
        scenario = SquareCrossingScenario(n_agents=24, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=True, horizontal=True, width=8, height=8)
    elif settings.scene == "20-corridor":
        scenario = CompositeScenarios([
            SquareCrossingScenario(n_agents=20, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=True, horizontal=False, width=8, height=8),
            SquareCrossingScenario(n_agents=20, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=False, horizontal=True, width=8, height=8)
        ])
    elif settings.scene == "24-corridor":
        scenario = CompositeScenarios([
            SquareCrossingScenario(n_agents=24, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=True, horizontal=False, width=8, height=8),
            SquareCrossingScenario(n_agents=24, agent_wrapper=agent_wrapper, min_distance=0.3, vertical=False, horizontal=True, width=8, height=8)
        ])    
    else:
        raise ValueError("Unrecognized scene: {}".format(settings.scene))
    env = Env(scenario=scenario, fps=1./config.STEP_TIME, timeout=config.VISUALIZATION_TIMEOUT, frame_skip=config.FRAME_SKIP,
        view=True
    )
    return env

def evaluate(ckpt_file):
    print(ckpt_file)
    print(settings.scene)

    ckpt = torch.load(ckpt_file, map_location="cpu")
    state_dict = {}
    for k, v in ckpt["model"].items():
        if "model.actor.model." in k:
            state_dict[k[18:]] = v
    model = ExpertNetwork(agent_dim=3, neighbor_dim=4, out_dim=2)
    model.load_state_dict(state_dict)
    if settings.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = settings.device
    model.to(device)

    env = env_wrapper(model)
    env.seed(0)
    model.eval()

    done, info = True, None
    trials = 0
    while True:
        if done:
            state = env.reset()
            env.figure.axes.set_title(os.path.join(os.path.basename(os.path.dirname(ckpt_file)), os.path.basename(ckpt_file)))
            t = time.time()
        else:
            state = state_
        act = [ag.act(s, env) for ag, s in zip(env.agents, state)]

        state_, rews, done, info = env.step(act)
        delay = config.STEP_TIME - time.time() + t
        if delay > 0:
            time.sleep(delay)
        t = time.time()
        if done:
            trials += 1
            time.sleep(2)
            if trials >= settings.max_trials:
                break


if __name__ == "__main__":
    if os.path.isfile(settings.ckpt):
        evaluate(settings.ckpt)
    else:
        def check(path):
            for f in sorted(os.listdir(path)):
                filename = os.path.join(path, f)
                if "ckpt" == f:
                    evaluate(filename)
                elif os.path.isdir(filename):
                    check(filename)
        check(settings.ckpt)
