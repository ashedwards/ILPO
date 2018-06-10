import gym
import numpy as np
import cv2
import gym_thor
import os
from baselines import deepq

STEPS = 500000
ENV = "ThorFridge-v0"
FILE = "final_models/thor/AB/"
BC_FILE = "final_models/thor/separate_actions/thor_bc.txt"
MODEL = "final_models/thor_model.pkl"


def main():
    if not os.path.exists("final_models/thor"):
        os.makedirs("final_models/thor")
        os.makedirs("final_models/thor/AB")
        os.makedirs("final_models/thor/separate_actions")

    env = gym.make(ENV)
    act = deepq.load(MODEL)
    action_file = open(BC_FILE, "w")
    steps = 0

    while steps < STEPS:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done and steps < STEPS:
            state_1 = cv2.resize(obs, (128, 128))

            if np.random.uniform(0,1) < .75:
                action =  act(obs[None])[0]
            else:
                action =  env.action_space.sample()

            obs, rew, done, _ = env.step(action)
            state_2 = cv2.resize(obs, (128, 128))

            cv2.imwrite(FILE + str(steps) + ".png", np.hstack([state_1, state_2]))
            action_file.write("[" + str(action) + "]\n")
            episode_rew += rew
            steps += 1

        print(steps)
        print("Episode reward", episode_rew)

    action_file.close()


if __name__ == '__main__':
    main()
