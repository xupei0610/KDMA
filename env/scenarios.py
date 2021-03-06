from typing import Callable, Sequence, Tuple
from .agents.base_agent import BaseAgent

import numpy

__all__ = [
    "CompositeScenarios",
    "CircleCrossingScenario",
    "SquareCrossingScenario",

    "CircleCrossing6Scenario",
    "CircleCrossing12Scenario"
]

class BaseScenario():
    def __init__(self, seed: int=None):
        self.rng = numpy.random.RandomState()
        self.seed(seed)

    def spawn(self):
        raise NotImplementedError

    def seed(self, s: int=None):
        self.custom_seed = s is not None
        self.rng.seed(s)

    def collide(self, agent0: BaseAgent, agent1: BaseAgent):
        if agent0.visible and agent1.visible:
            dist2 = (agent0.position.x-agent1.position.x)**2 + (agent0.position.y-agent1.position.y)**2
            return dist2 <= (agent0.radius+agent1.radius)**2
        return False

    def placeable(self, agent: BaseAgent):
        for o in self.agents:
            if self.collide(agent, o):
                return False
        return True

    def __iter__(self):
        self.counter = 0
        self.agents = []
        return self
    
    def __next__(self):
        agent = self.spawn()
        self.agents.append(agent)
        self.counter += 1
        return agent


class CompositeScenarios(BaseScenario):

    def __init__(self, scenarios: Sequence[BaseScenario], prob=None, seed: int=None):
        self.scenarios = [s for s in scenarios]
        self.prob = prob
        super().__init__(seed)
    
    def seed(self, seed):
        super().seed(seed)
        for i, s in enumerate(self.scenarios):
            if not s.custom_seed:
                s.rng = self.rng
            
    def __iter__(self):
        i = self.rng.choice(len(self.scenarios), p=self.prob)
        self.spawn = self.scenarios[i].spawn
        return self.scenarios[i].__iter__()


class CircleCrossingScenario(BaseScenario):

    def __init__(self,
        n_agents: int or Tuple[int, int],
        radius: float or Tuple[float, float],
        agent_wrapper: Callable[[], BaseAgent],
        noise: float = 0,
        min_distance: float = 0,
        seed: int = None
    ):
        self.n_agents = n_agents
        self.radius = radius
        self.agent_wrapper = agent_wrapper
        self.min_distance = min_distance
        self.noise = noise
        super().__init__(seed)
    
    def __iter__(self):
        if hasattr(self.n_agents, "__len__"):
            self._n_agents = self.rng.randint(self.n_agents[0], self.n_agents[1])
        else:
            self._n_agents = self.n_agents
        if hasattr(self.radius, "__len__"):
            self._radius = self.rng.random()*(self.radius[1]-self.radius[0]) + self.radius[0]
        else:
            self._radius = self.radius
        return super().__iter__()

    def spawn(self):
        if self.counter >= self._n_agents:
            raise StopIteration

        agent = self.agent_wrapper()
        r = agent.radius
        agent.radius += self.min_distance

        while True:
            a = self.rng.random() * 2*numpy.pi
            agent.position = numpy.cos(a)*self._radius, numpy.sin(a)*self._radius
            if self.noise:
                agent.position = (
                    agent.position.x + (self.rng.random()-0.5)*2*self.noise,
                    agent.position.y + (self.rng.random()-0.5)*2*self.noise
                )
            if self.placeable(agent): break
        agent.goal = -agent.position.x, -agent.position.y
        # agent.velocity = agent.preferred_velocity(0.12)
        agent.radius = r
        return agent


class SquareCrossingScenario(BaseScenario):
    def __init__(self,
        n_agents: int or Tuple[int, int],
        width: float or Tuple[float, float],
        height: float or Tuple[float, float],
        vertical: bool,
        horizontal: bool,
        agent_wrapper: Callable[[], BaseAgent],
        noise: float = 0,
        min_distance: float = 0,
        seed: int = None
    ):
        self.n_agents = n_agents
        self.width = width
        self.height = height
        self.vertical = vertical
        self.horizontal = horizontal
        self.agent_wrapper = agent_wrapper
        self.min_distance = min_distance
        super().__init__(seed)

    def __iter__(self):
        if hasattr(self.n_agents, "__len__"):
            self._n_agents = self.rng.randint(self.n_agents[0], self.n_agents[1])
        else:
            self._n_agents = self.n_agents
        if hasattr(self.width, "__len__"):
            self._width = self.rng.random()*(self.width[1]-self.width[0]) + self.width[0]
        else:
            self._width = self.width
        if hasattr(self.height, "__len__"):
            self._height = self.rng.random()*(self.height[1]-self.height[0]) + self.height[0]
        else:
            self._height = self.height
        self.goals = []
        return super().__iter__()
    

    def spawn(self):
        if self.counter >= self._n_agents:
            raise StopIteration

        agent = self.agent_wrapper()
        r = agent.radius
        agent.radius += self.min_distance
        r2 = agent.radius*agent.radius

        if self.vertical and self.horizontal:
            vertical = self.rng.random() > 0.5
        else:
            vertical = self.vertical
        while True:
            if vertical:
                x = self.rng.random()-0.5
                if self.horizontal: x *= 0.5
                y = (self.rng.random()-0.5) * 0.5 # (-0.25, 0.25)
                if y < 0:
                    y -= 0.25
                else:
                    y += 0.25
            else:
                x = (self.rng.random()-0.5) * 0.5
                if x < 0:
                    x -= 0.25
                else:
                    x += 0.25
                y = self.rng.random()-0.5
                if self.vertical: y *= 0.5
            agent.position = x * self._width, y * self._height
            if self.placeable(agent): break
        while True:
            if vertical:
                x = self.rng.random()-0.5
                if self.horizontal: x *= 0.5
                y = (self.rng.random()-0.5) * 0.5 # (-0.25, 0.25)
                if y < 0:
                    y -= 0.25
                else:
                    y += 0.25
                if ((agent.position.y > 0 and y > 0) or (agent.position.y < 0 and y < 0)):
                    y = -y
            else:
                x = (self.rng.random()-0.5) * 0.5
                if x < 0:
                    x -= 0.25
                else:
                    x += 0.25
                if (agent.position.x > 0 and x > 0) or (agent.position.x < 0 and x < 0):
                    x = -x
                y = self.rng.random()-0.5
                if self.vertical: y *= 0.5
            x *= self._width
            y *= self._height
            if (agent.position.x-x)**2 + (agent.position.y-y)**2 <= r2:
                continue
            placeable = True
            for gx, gy in self.goals:
                if (gx-x)**2 + (gy-y)**2 <= r2:
                    placeable = False
                    break
            if placeable:
                agent.goal = x, y
                break
        self.goals.append((agent.goal.x, agent.goal.y))
        agent.radius = r
        return agent



class PredefinedScenario(BaseScenario):
    def __init__(self,
        agent_wrapper: Callable[[], BaseAgent],
        scale : int = 1
    ):
        self.agent_wrapper = agent_wrapper
        super().__init__()
        self.scale = scale
    def spawn(self):
        if self.counter >= len(self.POS):
            raise StopIteration

        agent = self.agent_wrapper()
        agent.position = self.POS[self.counter][0]*self.scale, self.POS[self.counter][1]*self.scale
        agent.goal = self.GOAL[self.counter][0]*self.scale, self.GOAL[self.counter][1]*self.scale
        return agent


class CircleCrossing6Scenario(PredefinedScenario):

    POS = (
        ( 3.7104692, -1.5660178 ),
        ( 2.4903586, 3.3488475 ),
        ( -1.2998195, 3.90333 ),
        ( -4.0630183, 1.1232517 ),
        ( -2.9886541, -2.7252258 ),
        ( 0.58739246, -4.0228777 )
    )

    GOAL = (
        ( -3.9725885, 1.7843779 ),
        ( -2.7661666, -3.4184108 ),
        ( 1.4674827, -4.0561664 ),
        ( 4.2410774, -0.92563815 ),
        ( 3.1549813, 3.1627561 ),
        ( -1.0496976, 4.6132372 )
    )

    def __init__(self,
        agent_wrapper: Callable[[], BaseAgent],
        scale : int = 0.46716322317696163
    ):
        super().__init__(agent_wrapper, scale)
    


class CircleCrossing12Scenario(PredefinedScenario):

    POS = (
        (3.6143744, 1.1175725),
        (2.3853909, 2.2851516),
        (1.2247712, 3.8650695),
        (-0.4762151, 3.3848816),
        (-2.838231, 2.6234833),
        (-3.6877534, 0.9764592),
        (-3.396552, -1.4452063),
        (-2.0577695, -2.4942946),
        (-0.62386751, -4.0112653),
        (1.1290323, -3.4432024),
        (2.4359613, -2.1397746),
        (3.4712598, -0.83114835)
    )

    GOAL = (
        (-4.2847164, -1.5056181),
        (-3.2118745, -2.9733542),
        (-1.0373271, -3.7547732),
        (0.65222115, -4.2867512),
        (3.2604236, -2.8074883),
        (4.221533, -0.909147),
        (3.7665522, 1.9906672),
        (2.6808445, 3.550647),
        (1.2148837, 4.2657175),
        (-1.5079789, 4.1998218),
        (-3.1926017, 3.1203148),
        (-4.2111265, 1.0495945)
    )

    def __init__(self,
        agent_wrapper: Callable[[], BaseAgent],
        scale : int = 0.47624522851691475
    ):
        super().__init__(agent_wrapper, scale)
        
