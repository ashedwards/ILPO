"""Runs Behavioral Cloning by Observation (BCO)"""
from utils import *
from vector_bco import VectorBCO
from collections import deque
import gym
import cv2
import os
import random

states = []
next_states = []

STEPS = 10000

# Create actions for expert training set
for line in open(args.input_dir):
    state, next_state = line.replace("\n", "").split(" ")
    state = eval(state)
    next_state = eval(next_state)
    states.append(state)
    next_states.append(next_state)

class Policy(VectorBCO):
    def __init__(self, sess, shape,verbose=False, use_encoding=False, experiment=False, exp_writer=None):
        """Initializes the ILPO policy network."""

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

        self.action_label, self.loss = self.policy_net(self.state, self.action)

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

    def policy_net(self, state, action):
        """Network for remapping incorrect action labels."""

        with tf.variable_scope("action_remap"):
            state_encoding = lrelu(slim.flatten(self.create_encoder(state)[-1]), .2)
            real_action_one_hot = tf.one_hot(action, args.real_actions, dtype="int32")
            prediction = lrelu(fully_connected(state_encoding, 64), .2)
            prediction = lrelu(fully_connected(prediction, 32), .2)
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
        terminal = False
        total_reward = 0
        obs = game.reset()

        while not terminal:
            action = self.greedy(obs)
            obs, reward, terminal, _ = game.step(action)
            total_reward += reward

        if not self.experiment:
            reward_summary = sess.run([self.reward_summary], feed_dict={self.reward: [total_reward]})[0]
            self.summary_writer.add_summary(reward_summary, t)
        else:
            self.exp_writer.write(str(t) + "," + str(total_reward) + "\n")

    def run_policy(self, N, seed):
        """Run the policy."""

        game = gym.make(args.env)
        game.seed(seed)
        obs = game.reset()

        terminal = False
        D = deque()
        iteration = 0

        # While policy improvement
        while len(D) < 1000:
            # Evaluate policy.
            self.eval_policy(game, iteration)

            # Generate samples using pi
            for sample in range(0, 50):
                if terminal:
                    obs = game.reset()
                    steps = 0.

                prev_obs = np.copy(obs)

                if iteration == 0:
                    action = game.action_space.sample()
                else:
                    action = self.greedy(obs)

                obs, reward, terminal, _ = game.step(action)

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

                    if t % 1000 == 0:
                        print("Learning inverse model " + str(t) + ": " + str(loss) + " " + str(N))
                        print("Iteration", iteration)

            E = deque()

            # Generate set of agent-specific state transitions
            for expert_samples in range(0, len(states), 1000):
                state = states[expert_samples:expert_samples + 1000]
                next_state = next_states[expert_samples:expert_samples + 1000]

                # Use M to generate action
                action = np.argmax(self.inverse_action(state, next_state), axis=1)

                for batch in range(0, len(action)):
                    E.append((state[batch], action[batch], next_state[batch]))

                if expert_samples % 1000 == 0:
                    print("Creating expert sample " + str(expert_samples) + ": " + str(N))

            # Improve pi by behavioral cloning
            for t in range(0, STEPS):
                minibatch = random.sample(E, args.batch_size)
                obs_batch = [d[0] for d in minibatch]
                action_batch = [d[1] for d in minibatch]

                _, loss_summary, loss = sess.run([self.train_step, self.loss_summary, self.loss], feed_dict={
                    self.state: obs_batch,
                    self.action: action_batch})

                if not self.experiment:
                    self.summary_writer.add_summary(loss_summary, t)

                if t % 100 == 0:
                    print("Learning policy model " + str(t) + ": " + str(loss) + " " + str(N))

            iteration += 50


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
        ilpo = Policy(sess=sess, shape=[None, args.n_dims], use_encoding=False, verbose=False, experiment=True, exp_writer=exp_writer)
        ilpo.run_policy(0, exp)

    exp_writer.close()

