from typing import Callable, Sequence

import numpy

from .scenarios import BaseScenario
from .agents.base_agent import BaseAgent


class DecentralizedMultiAgentEnv():

    def __init__(self,
        scenario: BaseScenario,
        fps: int = 30,
        frame_skip: int = 1,
        timeout: int=None, 
        view: bool or float = False
    ):
        self.scenario = scenario

        self._fps = fps
        self.frame_skip = frame_skip
        self.timeout = timeout

        self.agents: Sequence[BaseAgent] = []

        self.figure = None
        self.markers = []
        self.trajectories = []
        self.view = view

        self.snap_counter = 0

    def seed(self, s):
        self.scenario.seed(s)

    def reset(self):
        self.agents.clear()
        for agent in self.scenario:
            self.agents.append(agent)
        
        self.simulation_steps = 0
        self.info = dict(collided_agents=set(), arrived_agents=set())

        for i in range(min(len(self.trajectories), len(self.markers))):
            self.trajectories[i][0].clear()
            self.trajectories[i][1].clear()
        if len(self.markers) > len(self.agents):
            for i in range(len(self.agents), len(self.markers)):
                self.markers[i][0].set_visible(False)
                self.markers[i][1].set_visible(False)
                self.markers[i][2].set_visible(False)
        if self.view:
            self.render(None if type(self.view) == bool else self.view)
        if self.figure:
            ax = self.figure.axes
            ax.set_xlim(0, 0.01)
            ax.set_ylim(0, 0.01)

        return self.observe()

    def step(self, action):
        self.simulation_steps += 1
        self.info["arrived_agents"].clear()

        for agent, a in zip(self.agents, action):
            if a is None:
                agent.accelerate = 0, 0
                agent.velocity = 0, 0
            else:
                agent.accelerate = a[0], a[1]
                agent.velocity = \
                    agent.velocity.x+agent.accelerate.x/self.fps, \
                    agent.velocity.y+agent.accelerate.y/self.fps
        for _ in range(self.frame_skip):
            for agent in self.agents:
                agent.position = \
                    agent.position.x+agent.velocity.x*self.step_time, \
                    agent.position.y+agent.velocity.y*self.step_time
            for i in range(len(self.agents)):
                for j in range(i+1, len(self.agents)):
                    if self.scenario.collide(self.agents[i], self.agents[j]):
                        self.info["collided_agents"].add(self.agents[i])
                        self.info["collided_agents"].add(self.agents[j])
                        self.agents[i].velocity = 0., 0.
                        self.agents[j].velocity = 0., 0.
            if self.view:
                self.render(None if type(self.view) == bool else self.view)
        for agent in self.agents:
            if agent in self.info["collided_agents"]: continue
            dist2 = (agent.position.x - agent.goal.x)**2 + (agent.position.y - agent.goal.y)**2
            if dist2 <= agent.radius*agent.radius:
                self.info["arrived_agents"].add(agent)
        
        obs = self.observe()
        rews = [self.reward(i, agent) for i, agent in enumerate(self.agents)]
        
        if len(self.info["collided_agents"]) + len(self.info["arrived_agents"]) < len(self.agents):
            terminal = False
        else:
            terminal = True
            for agent in self.agents:
                if agent not in self.info["collided_agents"] and agent not in self.info["arrived_agents"]:
                    terminal = False
                    break
        
        if not terminal and self.timeout and self.simulation_steps >= self.timeout:
            self.info["TimeLimit.truncated"] = True
            terminal = True

        if self.view:
            self.render(None if type(self.view) == bool else self.view)
        return obs, rews, terminal, self.info

    def observe(self):
        return [agent.observe(self) for agent in self.agents]
    
    def reward(self, idx, agent):
        raise NotImplementedError

    @property
    def step_time(self):
        return self._step_time

    @property
    def fps(self):
        return self._fps
    
    @fps.setter
    def fps(self, v):
        self._fps = v
        self._step_time = 1. / (self.fps*self.frame_skip)

    @property
    def frame_skip(self):
        return self._frame_skip
    
    @frame_skip.setter
    def frame_skip(self, v):
        self._frame_skip = v
        self._step_time = 1. / (self.fps*self.frame_skip)

    def render(self, view_size=None):
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap("cool" if len(self.agents) > 20 else "tab20", len(self.agents))

        if self.figure is None:
            plt.ion()
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            self.figure = ax.text(x=0.95, y=0.95, s="", fontsize=16,
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes)
            plt.tight_layout()

            self.show_goal = True
            self.show_trajectory = True

            def key_press_callback(event):
                if event.key == "t":
                    if not self.show_goal and not self.show_trajectory:
                        self.show_trajectory = True
                    elif not self.show_goal and self.show_trajectory:
                        self.show_goal = True
                    elif self.show_goal and self.show_trajectory:
                        self.show_trajectory = False
                        for i in range(min(len(self.agents), len(self.markers))):
                            self.markers[i][1].set_visible(False)
                    else: # self.show_goal and not self.show_trajectory
                        self.show_goal = False
                        for i in range(min(len(self.agents), len(self.markers))):
                            self.markers[i][2].set_visible(False)

            fig.canvas.mpl_connect("key_press_event", key_press_callback)
            print("Press t to toggle markers.")

        else:        
            fig = self.figure.figure
            ax = self.figure.axes

        fig.canvas.flush_events()

        if len(self.markers) < len(self.agents):
            n = len(self.agents) - len(self.markers)
            for _  in range(n):
                marker = plt.Circle((0, 0), 1, visible=False)
                line = plt.Line2D([], [], alpha=0.25, visible=False)
                goal = plt.Circle((0, 0), 1, visible=False)
                self.trajectories.append([[], []])
                self.markers.append((marker, line, goal))
                ax.add_artist(line)
                ax.add_artist(goal)
                ax.add_artist(marker)
        self.figure.set_text("Step: {}".format(self.simulation_steps))

        for i, (traj, art, agent) in enumerate(zip(self.trajectories, self.markers, self.agents)):
            marker, line, goal = art
            c = cmap(i)
            if agent in self.info["collided_agents"]:
                ec = "r"
                alpha=0.5
            elif agent in self.info["arrived_agents"]:
                ec = c
                alpha= 0.75
            else:
                if not agent.visible: continue
                ec = None
                alpha=1.0
            traj[0].append(agent.position.x)
            traj[1].append(agent.position.y)
            if self.show_trajectory :
                line.set_data(traj)
                line.update(dict(color=c, visible=True))
            if self.show_goal:
                goal.update(dict(
                    center=(agent.goal.x, agent.goal.y),
                    radius=agent.radius,
                    color=c, alpha=0.2,
                    visible=True))
            marker.update(dict(
                center=(agent.position.x, agent.position.y),
                radius=agent.radius, 
                color=c, ec=ec, alpha=alpha,
                label=i, visible=True))
        
        if view_size:
            view_size = numpy.multiply(view_size, [[1,1],[1,1]])
            if view_size[0][0] == view_size[0][1]:
                ax.set_xlim(-view_size[0][0], view_size[0][1])
            else:
                ax.set_xlim(view_size[0][0], view_size[0][1])
            if view_size[1][0] == view_size[1][1]:
                ax.set_ylim(-view_size[1][0], view_size[1][1])
            else:
                ax.set_ylim(view_size[1][0], view_size[1][1])
        else:
            min_x, max_x = numpy.inf, -numpy.inf
            min_y, max_y = numpy.inf, -numpy.inf
            for agent in self.agents:
                min_x = min([min_x, agent.position.x, agent.goal.x])
                max_x = max([max_x, agent.position.x, agent.goal.x])
                min_y = min([min_y, agent.position.y, agent.goal.y])
                max_y = max([max_y, agent.position.y, agent.goal.y])
            min_x_, max_x_ = ax.get_xlim()
            min_y_, max_y_ = ax.get_ylim()
            if min_x_ > min_x or max_x_ < max_x or min_y_ > min_y or max_y_ < max_y:
                min_x = min(min_x, min_x_)
                max_x = max(max_x, max_x_)
                min_y = min(min_y, min_y_)
                max_y = max(max_y, max_y_)
                min_ = min(min_x, min_y)
                max_ = max(max_x, max_y)
                mid = (min_+max_)*0.5
                min_ = mid - (mid-min_)*1.25
                max_ = mid + (max_-mid)*1.25
                
                ax.set_xlim(min_, max_)
                ax.set_ylim(min_, max_)


        fig.canvas.draw()

