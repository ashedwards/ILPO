import gym

from baselines import deepq


def main():
    env = gym.make("Acrobot-v0")
    model = deepq.models.mlp([64])
    act = deepq.learn(
        env,
        q_func=model,
        lr=1e-3,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
    )
    print("Saving model to final_models/acrobot_model.pkl")
    act.save("final_models/acrobot_model.pkl")


if __name__ == '__main__':
    main()
