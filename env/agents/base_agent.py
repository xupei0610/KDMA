from __future__ import annotations
from typing import Sequence, Optional, Tuple

import math
from .utils import Vec2

class BaseAgent():
    def __init__(self,*,
        radius: float=0.1, visible: bool=True, observe_angle: float=2*math.pi, observe_radius: float=5,
        position: Vec2=None, velocity: Vec2=None, goal: Vec2=None,
        preferred_speed:float=None, max_speed:float=2.5, max_accelerate:float=None

    ):
        self._goal = Vec2(x=0, y=0) if goal is None else goal
        self._position = Vec2(x=0, y=0) if position is None else position
        self._velocity = Vec2(x=0, y=0) if velocity is None else velocity
        self._accelerate = Vec2(x=0, y=0)

        self.max_accelerate = max_accelerate
        self.max_speed = max_speed
        self.preferred_speed = preferred_speed

        self.radius = radius 
        self.visible = visible

        self.observe_angle = math.pi*2
        self.observe_radius = observe_radius

    @property
    def speed(self):
        return (self.velocity.x*self.velocity.x + self.velocity.y*self.velocity.y)**0.5
    
    @property
    def velocity(self):
        return self._velocity
    
    @velocity.setter
    def velocity(self, vals):
        self._velocity._x = vals[0]
        self._velocity._y = vals[1]
        if self.max_speed:
            norm2 = self.velocity.x*self.velocity.x + self.velocity.y*self.velocity.y
            if norm2 > self._max_speed2:
                scale = self._max_speed/(norm2**0.5)
                self._velocity._x *= scale
                self._velocity._y *= scale

    @property
    def max_speed(self):
        return self._max_speed
    
    @max_speed.setter
    def max_speed(self, v):
        if v:
            self._max_speed = abs(v)
            self._max_speed2 = v*v
            self.velocity = self._velocity
        else:
            self._max_speed = self._max_speed2 = None

    @property
    def accelerate(self):
        return self._accelerate
    
    @accelerate.setter
    def accelerate(self, vals):
        self._accelerate._x = vals[0]
        self._accelerate._y = vals[1]
        if self.max_accelerate:
            norm2 = self.accelerate.x*self.accelerate.x + self.accelerate.y*self.accelerate.y
            if norm2 > self._max_accelerate2:
                scale = self._max_accelerate/(norm2**0.5)
                self._accelerate._x *= scale
                self._accelerate._y *= scale

    @property
    def max_accelerate(self):
        return self._max_accelerate
    
    @max_accelerate.setter
    def max_accelerate(self, v):
        if v:
            self._max_accelerate = abs(v)
            self._max_accelerate2 = v*v
            self.accelerate = self._accelerate
        else:
            self._max_accelerate = self._max_accelerate2 = None
    
    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, vals):
        self._position._x, self._position._y = vals[0], vals[1]

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, vals):
        self._goal._x, self._goal._y = vals[0], vals[1]

    @property
    def observe_angle(self):
        return self._half_observe_angle*2

    @observe_angle.setter
    def observe_angle(self, ang):
        if ang >= 2*math.pi or ang <= -2*math.pi:
            self._half_observe_angle = math.pi
        else:
            ang = ang % (2*math.pi)
            self._half_observe_angle = 0.5*ang

    def observable(self, agent):
        if agent is self or not agent.visible:
            return False
        if self.observe_radius is None or self.observe_radius <= 0:
            return True
        dx = agent.position.x - self.position.x
        dy = agent.position.y - self.position.y
        dist2 = dx*dx + dy*dy
        if dist2 > self.observe_radius*self.observe_radius:
            return False
        if self._half_observe_angle < math.pi:
            dist = dist2**0.5
            dx, dy = dx/dist, dy/dist
            speed = self.accelerate.x*self.accelerate.x + \
                self.accelerate.y*self.accelerate.y
            speed = speed**0.5
            heading_x = self.accelerate.x/speed
            heading_y = self.accelerate.y/speed
            a = math.acos(dx*heading_x + dy*heading_y)
            if a > self._half_observe_angle:
                return False
        return True

    def preferred_velocity(self, dt:float, *, start: Optional[Tuple[float, float]] = None, dest: Optional[Tuple[float, float]] = None):
        if dest is None:
            if start is None:
                dp = self.goal.x - self.position.x, self.goal.y - self.position.y
            else:
                dp = self.goal.x - start[0], self.goal.y - start[1]
        else:
            if start is None:
                dp = dest[0] - self.position.x, dest[1] - self.position.y
            else:
                dp = dest[0] - start[0], dest[1] - start[1]
        v_pref = [dp[0]/dt, dp[1]/dt]

        if self.preferred_speed:
            speed = (v_pref[0]*v_pref[0] + v_pref[1]*v_pref[1])**0.5
            if speed > self.preferred_speed:
                scale = self.preferred_speed / speed
                v_pref[0] *= scale
                v_pref[1] *= scale
        elif self.max_speed:
            speed = (v_pref[0]*v_pref[0] + v_pref[1]*v_pref[1])**0.5
            if speed > self.max_speed:
                scale = self.max_speed / speed
                v_pref[0] *= scale
                v_pref[1] *= scale
        return v_pref
        
    def observe(self, env):
        return []
        raise NotImplementedError

    def act(self, state, env) -> Tuple[float, float] or None:
        if state is None: return state
        raise NotImplementedError
