"""Runs a trained BC policy."""
from utils import *
from image_bc import ImageBC
from collections import deque
import gym
import gym_thor
import cv2
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class Policy(ImageBC):
    def __init__(self, sess, shape, verbose=False):
        """Initializes the BC network."""

        self.sess = sess
        self.verbose = verbose
        self.inputs = tf.placeholder("float", shape)
        self.targets = tf.placeholder("float", [None])
        self.reward = tf.placeholder("float", [None])

        processed_inputs = self.process_inputs(self.inputs)
        processed_targets = self.targets

        self.model = self.create_model(processed_inputs, processed_targets)

        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(args.checkpoint)
        saver.restore(sess, checkpoint)

    def P(self, state):
        """Returns the next_state probabilities for a state."""

        return sess.run(self.model.actions, feed_dict={self.inputs: [state]})[0]

    def greedy(self, state):
        """Returns the greedy remapped action for a state."""

        action = np.argmax(self.P(state))

        if self.verbose:
            print(action)

        return action

    def eval_policy(self, game, t):
        """Evaluate the policy."""

        terminal = False
        total_reward = 0
        obs = game.reset()

        while not terminal:
          #  game.render()
            action = self.greedy(obs)
            obs, reward, terminal, _ = game.step(action)
            total_reward += reward

        print(total_reward)
        return total_reward

    def run_policy(self):
        """Run the policy."""

        game = gym.make(args.env)
        obs = game.reset()
        average_reward = 0

        for t in range(0, 100):
            average_reward += self.eval_policy(game, t)

        print("Average reward", average_reward / 100.)


sess = tf.Session()
with sess.as_default():
    bc = Policy(sess=sess, shape=[None, 128, 128, 3], verbose=False)
    bc.run_policy()

