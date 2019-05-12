"""Runs a trained ILPO policy in an online manner and concurrently fixes action inconsistencies."""
import gym
env = gym.make("CartPole-v1")
#env.render()

from utils import *
from image_ilpo import ImageILPO
from collections import deque
import gym
import cv2
import os
import coinrun.main_utils as utils
from coinrun import setup_utils, policies, wrappers, ppo2
from coinrun.config import Config
#from gym.envs.classic_control import rendering

utils.setup_mpi_gpus()
setup_utils.setup_and_load()
game = utils.make_general_env(1)
game = wrappers.add_final_wrappers(game)
game.reset()

args.checkpoint = 'final_coin_ilpo'
args.exp_dir = 'results/final_coin'
args.n_actions = 4
args.batch_size = 100
args.real_actions = 7
args.policy_lr = .001
args.ngf = 15
EXPLORE = 1000
FINAL_EPSILON = .2 # final value of epsilon
INITIAL_EPSILON = .2 # starting value of epsilon
COLLECT = 0

class Policy(ImageILPO):
    def __init__(self, sess, shape, verbose=True, use_encoding=False, experiment=False, exp_writer=None):
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
        self.exp_writer = exp_writer
        self.experiment = experiment

        processed_inputs = self.process_inputs(self.inputs)
        processed_targets = self.process_inputs(self.targets)
        processed_state = self.process_inputs(self.state)

        self.model = self.create_model(processed_inputs, processed_targets)

        self.state_encoding = self.encode(processed_inputs)[-1]
        self.action_label, self.loss = self.action_remap_net(processed_state, self.action, self.fake_action)
        self.loss_summary = tf.summary.scalar("policy_loss", tf.squeeze(self.loss))

        if not self.experiment:
            self.reward_summary = tf.summary.scalar("reward", tf.squeeze(self.reward[0]))
            self.summary_writer = tf.summary.FileWriter("policy_logs", graph=tf.get_default_graph())

        self.train_step = tf.train.AdamOptimizer(args.policy_lr).minimize(self.loss)

        ilpo_var_list = []
        policy_var_list = []

        # Restore ILPO params and initialize policy params.
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            if "ilpo" in var.name:
                ilpo_var_list.append(var)
            else:
                policy_var_list.append(var)

        saver = tf.train.Saver(var_list=ilpo_var_list)
        checkpoint = tf.train.latest_checkpoint(args.checkpoint)
        saver.restore(sess, checkpoint)
        sess.run(tf.variables_initializer(policy_var_list))

        self.deprocessed_outputs = [tf.image.convert_image_dtype(deprocess(output), dtype=tf.uint8, saturate=True) for output in self.model.outputs]
        #self.viewer = rendering.SimpleImageViewer()

    def min_action(self, state, action, next_state):
        """Find the minimum action for training."""

        # Given state and action, find the closest predicted next state to the real one.
        # Use the real action as a training label for remapping the action label.
        fake_next_states = sess.run(self.deprocessed_outputs, feed_dict={self.inputs: [state]})

        if self.use_encoding:
            next_state_encoding = sess.run(self.state_encoding, feed_dict={self.inputs: [next_state]})
            fake_state_encodings = [sess.run(
                self.state_encoding,
                feed_dict={self.inputs: fake_next_state}) for fake_next_state in fake_next_states]
            distances = [np.linalg.norm(next_state_encoding - fake_state_encoding) for fake_state_encoding in fake_state_encodings]
        else:
            distances = [np.linalg.norm(next_state - fake_next_state) for fake_next_state in fake_next_states]
        min_action = np.argmin(distances)
        min_state = fake_next_states[min_action][0]

        if self.verbose:
            display_states = [cv2.cvtColor(cv2.resize(fake_next_state[0], (128, 128)), cv2.COLOR_RGB2BGR) for fake_next_state in fake_next_states]
            cv2.imshow("outputs", np.hstack(display_states))
            cv2.imshow("state", cv2.cvtColor(cv2.resize(state, (128, 128)), cv2.COLOR_RGB2BGR))
            cv2.imshow("NextPrediction", np.hstack([
                cv2.cvtColor(cv2.resize(next_state, (128, 128)), cv2.COLOR_RGB2BGR),
                cv2.cvtColor(cv2.resize(min_state, (128, 128)), cv2.COLOR_RGB2BGR)]))
            cv2.waitKey(0)
        return min_action

    def action_remap_net(self, state, action, fake_action):
        """Network for remapping incorrect action labels."""

       # fake_state_encoding = tf.stop_gradient(slim.flatten(lrelu(self.encode(state)[-1], .2)))
        with tf.variable_scope("action_remap"):
            fake_state_encoding = lrelu(slim.flatten(self.create_encoder(state)[-1]), .2)
            fake_action_one_hot = tf.one_hot(fake_action, args.n_actions)
            fake_action_one_hot = lrelu(fully_connected(fake_action_one_hot, int(fake_state_encoding.shape[-1])), .2)
            real_action_one_hot = tf.one_hot(action, args.real_actions, dtype="float32")
            fake_state_action = tf.concat([fake_state_encoding, fake_action_one_hot], axis=-1)
            prediction = lrelu(fully_connected(fake_state_action, 64), .2)
            prediction = lrelu(fully_connected(prediction, 64), .2)
            prediction = fully_connected(prediction, args.real_actions)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=real_action_one_hot, logits=prediction))

            return tf.nn.softmax(prediction), loss

    def P(self, state):
        """Returns the next_state probabilities for a state."""

        return sess.run(self.model.actions, feed_dict={self.inputs: [state]})[0]

    def greedy(self, state):
        """Returns the greedy remapped action for a state."""

        p_state = self.P(state)
        action = np.argmax(p_state)

        remapped_action = self.sess.run(self.action_label, feed_dict={self.state: [state], self.fake_action: [action]})[0]

        if self.verbose:
            print(self.P(state))
            print(remapped_action)
            print("\n")

        return np.argmax(remapped_action)

    def render(self, obs):
        pass
        #self.viewer.imshow(obs)

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

                if np.random.uniform(0,1) <= .9:
                    action = self.greedy(obs)
                else:
                    action = game.action_space.sample()

                for _ in range(0, 1):
                    obs, reward, terminal, _ = game.step(np.array([action]))

                total_reward += reward
                steps +=1

            print("Average reward", total_reward / 10.)

        if not self.experiment:
            reward_summary = sess.run([self.reward_summary], feed_dict={self.reward: total_reward / 10.})[0]
            self.summary_writer.add_summary(reward_summary, t)
        else:
            self.exp_writer.write(str(t) + "," + str(total_reward / 10.) + "\n")

    def run_policy(self):
        """Run the policy."""
        terminal = [False]
        obs = game.reset()
        obs = np.squeeze(obs)
        obs = cv2.resize(obs, (128, 128))

        total_reward = 0
        D = deque()
        steps = 0
        episode = 0
        epsilon = INITIAL_EPSILON

        prev_obs = obs.copy()

        for t in range(0, 2200):
            self.render(obs)

            if t % 200 == 0 and t >= COLLECT:
                #print("Evaluating", t)
                self.eval_policy(game, t)
                obs = np.squeeze(game.reset())
                obs = cv2.resize(obs, (128, 128))

            if terminal[0]:
                terminal = [False]
                obs = np.squeeze(obs)
                obs = cv2.resize(obs, (128, 128))

                total_reward = 0

                if t > COLLECT:
                    episode += 1

            prev_obs = np.copy(obs)

            if np.random.uniform(0,1) > epsilon and t > COLLECT:
                action = self.greedy(obs)
            else:
                action = game.action_space.sample()

            if epsilon > FINAL_EPSILON and t >= COLLECT:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            for _ in range(0, 1):
                obs, reward, terminal, _ = game.step(np.array([action]))
            obs = np.squeeze(obs)
            obs = cv2.resize(obs, (128, 128))
            total_reward += reward
            fake_action = self.min_action(prev_obs, action, obs)

            D.append((prev_obs, action, fake_action))

            if len(D) >= args.batch_size and t > COLLECT:
                minibatch = random.sample(D, args.batch_size)
                obs_batch = [d[0] for d in minibatch]
                action_batch = [d[1] for d in minibatch]
                fake_action_batch = [d[2] for d in minibatch]

                _, loss_summary, loss = sess.run([self.train_step, self.loss_summary, self.loss], feed_dict={
                    self.state: obs_batch,
                    self.action: action_batch,
                    self.fake_action: fake_action_batch})

                print("Loss", loss)
                if not self.experiment:
                    self.summary_writer.add_summary(loss_summary, t)

            print(t)
            print("Epsilon", epsilon)


if not os.path.exists(args.exp_dir):
    os.makedirs(args.exp_dir)

for exp in range(0, 50):
    np.random.seed(exp)
    tf.set_random_seed(exp)
    random.seed(exp)
    print("Running experiment", exp)

    exp_writer = open(args.exp_dir + "/" + str(exp) + ".csv", "w")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    print("Running experiment", exp)

    tf.reset_default_graph()
    sess = tf.Session(config=config)


    with sess.as_default():
        ilpo = Policy(sess=sess, shape=[None, 128, 128, 3], verbose=False, experiment=True, exp_writer=exp_writer, use_encoding=True)
        ilpo.run_policy()
        exp_writer.close()

