import gym
import numpy as np
from baselines import deepq


BEST = 1
DEFAULT = .75
RANDOM = DEFAULT

ENV = "CartPole-v1"
FILE = "final_models/cartpole/cartpole.txt"
BC_FILE = "final_models/cartpole_bc/cartpole_bc.txt"
MODEL = "final_models/cartpole_model.pkl"


def main():
    env = gym.make(ENV)
    act = deepq.load(MODEL)
    steps = 0
    outfile = open(FILE, 'w')
    bcfile = open(BC_FILE, 'w')
    total_reward = 0
    episodes = 0

    while steps < 50000:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
           # env.render()
            state_1 = obs

            if np.random.uniform(0,1) <= RANDOM:
                action = act(obs[None])[0]
            else:
                action = env.action_space.sample()

            obs, rew, done, _ = env.step(action)
            state_2 = obs

            if RANDOM == DEFAULT:
                # write to AON file
                to_write = '['
                for w in state_1:
                    to_write += str(w) + ','
                to_write = to_write[:-1]
                to_write += ']'

                outfile.write(to_write)
                outfile.write(" ")
                to_write = '['

                for w in state_2:
                    to_write += str(w) + ','
                to_write = to_write[:-1]
                to_write += ']'

                outfile.write(to_write)
                outfile.write("\n")

                # write to BC file
                to_write = '['
                for w in state_1:
                    to_write += str(w) + ','
                to_write = to_write[:-1]
                to_write += ']'

                bcfile.write(to_write)
                bcfile.write(" ")

                bcfile.write("[" + str(action) + "]")
                bcfile.write(" ")

                to_write = '['
                for w in state_2:
                    to_write += str(w) + ','
                to_write = to_write[:-1]
                to_write += ']'

                bcfile.write(to_write)
                bcfile.write("\n")


            episode_rew += rew

            steps += 1

        print(steps)
        print("Episode reward", episode_rew)
        total_reward += episode_rew
        episodes += 1.

    print("Average reward", total_reward / episodes)
    outfile.close()
    bcfile.close()

if __name__ == '__main__':
    main()
