import ai2thor.controller
import numpy as np
import cv2
import gym
from gym import spaces

# Kitchens: FloorPlan1 - FloorPlan30
# Living rooms: FloorPlan201 - FloorPlan230
# Bedrooms: FloorPlan301 - FloorPlan330
# Bathrooms: FloorPLan401 - FloorPlan430


class ThorEnv(gym.Env):
    def __init__(self):
        self.controller = ai2thor.controller.Controller()
        self.controller.start()
        self.controller.reset('FloorPlan2')
        self.controller.step(dict(action='Initialize', gridSize=0.25))

        self.actions = ["MoveAhead", "MoveBack", "RotateLeft"]
        self.controller.step(dict(action='Teleport', x=1, y=.98, z=3.25))
        self.object_name = 'Fridge'

        self.action_space = spaces.Discrete(3)
        self.obs_shape = [128, 128, 3]
        self.observation_space = spaces.Box(low=0, high=255, shape=self.obs_shape)
        self.min_distance = 1.005
        self.steps = 0
        self.max_steps = 100

    def _seed(self, x):
        return x

    def _step(self, a):
        event = self.controller.step(dict(action=self.actions[a]))

        # Find how close agent is to object distance.
        for obj in event.metadata['objects']:
            if self.object_name in obj['name']:
                distance = obj['distance']
                break

        if distance <= self.min_distance:
            done = True
        elif self.steps >= self.max_steps:
            done = True
        else:
            done = False

        reward = -distance

        self.steps += 1

        return self.get_observation(event), reward, done, {}

    def _reset(self):
        event = self.controller.step(dict(action='Teleport', x=1, y=.98, z=3.25))
        self.steps = 0
        return self.get_observation(event)

    def get_observation(self, event):
        return cv2.resize(event.cv2image(), (128, 128))





