"""Runs Behavioral Cloning by Observation (BCO)"""
import gym
env = gym.make("CartPole-v1")
#env.render()

from utils import *
from collections import deque
import gym
import cv2
import os
import coinrun.main_utils as utils
from coinrun import setup_utils, policies, wrappers, ppo2
from coinrun.config import Config
#from gym.envs.classic_control import rendering
from collections import deque
import random
from image_bco import ImageBCO

utils.setup_mpi_gpus()
setup_utils.setup_and_load()
game = utils.make_general_env(1)
game = wrappers.add_final_wrappers(game)
game.reset()

args.checkpoint = 'coin_ilpo'
args.input_dir = 'final_models/coin'
args.exp_dir = 'results/final_coin_bco'
args.n_actions = 4
args.real_actions = 4
args.policy_lr = .0001
args.batch_size = 100
args.ngf = 15
states = []
next_states = []
FINAL_EPSILON = .2 # final value of epsilon
INITIAL_EPSILON = .2 # starting value of epsilon
EXPLORE = 1000
STEPS = 500
E = deque()
expert_samples = 0
# Create actions for expert training set

def process_obs(image_file):
    image = cv2.imread(image_file)
    state = image[:,0:128,:]

    next_state = image[:,128:256,:]

    return state, next_state

for image_file in glob.glob(args.input_dir + "/*.png"):
    E.append((process_obs(image_file)))
    print(len(E))


class Policy(ImageBCO):
    def __init__(self, sess, shape,verbose=False, use_encoding=False, experiment=False, exp_writer=None):
        """Initializes the ILPO policy network."""
       # self.viewer = rendering.SimpleImageViewer()
        self.sess = sess
        self.verbose = verbose
        self.use_encoding = use_encoding
        self.inputs = tf.placeholder("float", shape)
        self.targets = tf.placeholder("float", shape)
        self.state = tf.placeholder("float", shape)
        self.action = tf.placeholder("int32", [None])
        self.fake_action = tf.placeholder("int32", [None])
        self.reward = tf.placeholder("float", [None])
        self.experiment = experiment
        self.exp_writer = exp_writer

        processed_inputs = self.process_inputs(self.inputs)
        processed_targets = self.process_inputs(self.targets)
        processed_state = self.process_inputs(self.state)

        self.model = self.create_model(processed_inputs, self.action, processed_targets)

        self.action_label, self.loss = self.policy_net(processed_state, self.action)

        self.loss_summary = tf.summary.scalar("policy_loss", tf.squeeze(self.loss))
        self.forward_summary =  tf.summary.scalar("forward_loss", tf.squeeze(self.model.gen_loss_L1))

        if not experiment:
            self.reward_summary = tf.summary.scalar("reward", self.reward[0])
            self.summary_writer = tf.summary.FileWriter("policy_logs", graph=tf.get_default_graph())

        self.train_step = tf.train.AdamOptimizer(args.policy_lr).minimize(self.loss)
        self.inverse_train_step = tf.train.AdamOptimizer(args.lr).minimize(self.model.gen_loss_L1)

        sess.run(tf.global_variables_initializer())

    def inverse_action(self, state, next_state):
        """Find the minimum action for training."""

        return sess.run(self.model.actions, feed_dict={self.inputs: state, self.targets: next_state})

    def create_small_encoder(self, state):
        """Creates state embedding."""

        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = conv(state, args.ngf, stride=2)
            layers.append(output)

        layer_specs = [
            args.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = conv(rectified, out_channels, stride=2)
                layers.append(convolved)

        return layers

    def policy_net(self, state, action):
        """Network for remapping incorrect action labels."""

        with tf.variable_scope("action_remap"):
            state_encoding = lrelu(slim.flatten(self.create_small_encoder(state)[-1]), .2)
            real_action_one_hot = tf.one_hot(action, args.real_actions, dtype="int32")
            prediction = lrelu(fully_connected(state_encoding, 64), .2)
            prediction = lrelu(fully_connected(prediction, 64), .2)
            prediction = fully_connected(prediction, args.real_actions)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=real_action_one_hot, logits=prediction))

            return tf.nn.softmax(prediction), loss

    def greedy(self, state):
        """Returns the greedy remapped action for a state."""

        # TODO: random action selection
        remapped_action = self.sess.run(self.action_label, feed_dict={self.state: [state]})[0]

        if self.verbose:
            print(remapped_action)

        return np.argmax(remapped_action)

    def eval_policy(self, game, t):
        """Evaluate the policy."""

        total_reward = 0

        for evaluation in range(0, 10):
            terminal = [False]
            obs = np.squeeze(game.reset())
            steps = 0
            print("Evaluating", evaluation)

            while not terminal[0] and steps < 200:
                obs = np.squeeze(obs)
                self.render(obs)
                obs = cv2.resize(obs, (128, 128))
                if np.random.uniform(0,1) < .9:
                    action = self.greedy(obs)
                else:
                    action = game.action_space.sample()
                obs, reward, terminal, _ = game.step(np.array([action]))

                total_reward += reward
                steps +=1

            print("Average reward", total_reward / 10.)
        if not self.experiment:
            reward_summary = sess.run([self.reward_summary], feed_dict={self.reward: total_reward / 10.})[0]
            self.summary_writer.add_summary(reward_summary, t)
        else:
            self.exp_writer.write(str(t) + "," + str(total_reward / 10.) + "\n")

    def render(self, obs):
        pass
        #self.viewer.imshow(obs)


    def run_policy(self, N, seed):
        """Run the policy."""

        obs = np.squeeze(game.reset())
        obs = cv2.resize(obs, (128, 128))

        terminal = [False]
        D = deque()
        iteration = 0
        prev_obs = obs.copy()

        epsilon = INITIAL_EPSILON

        # While policy improvement
        while len(D) <= 2000:
            # Evaluate policy.
            self.eval_policy(game, iteration)
            terminal = [True]

            # Generate samples using pi
            for sample in range(0, 200):
                if terminal[0]:
                    obs = np.squeeze(game.reset())
                    obs = cv2.resize(obs, (128, 128))
                    steps = 0.

                prev_obs = np.copy(obs)

                if iteration == 0 or np.random.uniform(0,1) > epsilon:
                    action = self.greedy(obs)
                else:
                    action = game.action_space.sample()

                if epsilon > FINAL_EPSILON:
                    epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

                obs, reward, terminal, _ = game.step(np.array([action]))
                obs = np.squeeze(obs)
                obs = cv2.resize(obs, (128, 128))

                # Append samples
                D.append((prev_obs, action, obs))

            # Improve M (inverse dynamics model) by model learning
            for t in range(0, STEPS):
                if len(D) >= args.batch_size:
                    minibatch = random.sample(D, args.batch_size)
                    obs_batch = [d[0] for d in minibatch]
                    action_batch = [d[1] for d in minibatch]
                    target_batch = [d[2] for d in minibatch]

                    _, loss = sess.run([self.inverse_train_step, self.model.gen_loss_L1], feed_dict={
                        self.inputs: obs_batch,
                        self.action: action_batch,
                        self.targets: target_batch})

                    if t % 100 == 0:
                        print("Learning inverse model " + str(t) + ": " + str(loss) + " " + str(N))
                        print("Iteration", len(D))

            # Improve pi by behavioral cloning
            for t in range(0, STEPS):
                minibatch = random.sample(E, args.batch_size)
                obs_batch = [d[0] for d in minibatch]
                next_obs_batch = [d[1] for d in minibatch]

                action_batch = np.argmax(self.inverse_action(obs_batch, next_obs_batch), axis=1)

                _, loss_summary, loss = sess.run([self.train_step, self.loss_summary, self.loss], feed_dict={
                    self.state: obs_batch,
                    self.action: action_batch})

                if not self.experiment:
                    self.summary_writer.add_summary(loss_summary, t)

                if t % 100 == 0:
                    print("Learning policy model " + str(t) + ": " + str(loss) + " " + str(N))

            iteration += 200


if not os.path.exists(args.exp_dir):
    os.makedirs(args.exp_dir)

for exp in range(0, 50):
    exp_writer = open(args.exp_dir + "/" + str(exp) + ".csv", "w")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    print("Running experiment", exp)

    tf.reset_default_graph()
    sess = tf.Session(config=config)

    np.random.seed(exp)
    tf.set_random_seed(exp)
    random.seed(exp)

    with sess.as_default():
        ilpo = Policy(sess=sess, shape=[None, 128, 128, 3], use_encoding=False, verbose=False, experiment=True, exp_writer=exp_writer)
        ilpo.run_policy(0, exp)
        exp_writer.close()

