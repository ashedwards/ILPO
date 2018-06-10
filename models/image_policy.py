"""Runs a trained ILPO policy in an online manner and concurrently fixes action inconsistencies."""
from utils import *
from image_ilpo import ImageILPO
from collections import deque
import gym
import gym_thor
import gym_ple
import cv2
import os

game = gym.make(args.env)

class Policy(ImageILPO):
    def __init__(self, sess, shape, input_action, verbose=True, use_encoding=True, experiment=False, exp_writer=None):
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

        self.model = self.create_model(processed_inputs, processed_targets, input_action=input_action)
        self.action_label, loss = self.action_remap_net(self.state, self.action, self.fake_action)


        self.loss_summary = tf.summary.scalar("policy_loss", tf.squeeze(loss))

        if not self.experiment:
            self.reward_summary = tf.summary.scalar("reward", tf.squeeze(self.reward[0]))
            self.summary_writer = tf.summary.FileWriter("policy_logs", graph=tf.get_default_graph())

        self.train_step = tf.train.AdamOptimizer(args.policy_lr).minimize(loss)

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

        self.state_encoding = self.encode(processed_inputs)
        self.deprocessed_outputs = [tf.image.convert_image_dtype(deprocess(output), dtype=tf.uint8, saturate=True) for output in self.model.outputs]

    def min_action(self, state, action, next_state):
        """Find the minimum action for training."""

        # Given state and action, find the closest predicted next state to the real one.
        # Use the real action as a training label for remapping the action label.
        fake_next_states = sess.run(self.deprocessed_outputs, feed_dict={self.inputs: [state]})

        if self.use_encoding:
            next_state_encoding = sess.run(self.state_encoding, feed_dict={self.inputs: [next_state]})[0]
            fake_state_encodings = [sess.run(
                self.state_encoding,
                feed_dict={self.inputs: fake_next_state})[0] for fake_next_state in fake_next_states]
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

        with tf.variable_scope("action_remap"):
            fake_state_encoding = lrelu(slim.flatten(self.create_encoder(state)[-1]), .2)
            fake_action_one_hot = tf.one_hot(fake_action, args.n_actions)
            fake_action_one_hot = lrelu(fully_connected(fake_action_one_hot, int(fake_state_encoding.shape[-1])), .2)
            real_action_one_hot = tf.one_hot(action, args.real_actions, dtype="int32")
            fake_state_action = tf.concat([fake_state_encoding, fake_action_one_hot], axis=-1)
            prediction = lrelu(fully_connected(fake_state_action, 64), .2)
            prediction = lrelu(fully_connected(prediction, 64), .2)
            prediction = fully_connected(prediction, args.real_actions)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=real_action_one_hot, logits=prediction))

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

    def run_policy(self):
        """Run the policy."""

        obs = game.reset()

        terminal = False
        obs = cv2.resize(obs, (128, 128))
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)

        total_reward = 0
        D = deque()
        steps = 0

        for t in range(0, 10000):
            if len(D) > 50000:
                D.popleft()

            if terminal:
                obs = game.reset()
                obs = cv2.resize(obs, (128, 128))
                obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)

                total_reward = 0

            prev_obs = np.copy(obs)

            if np.random.uniform(0,1) < .8:
                action = self.greedy(obs)
            else:
                action = game.action_space.sample()

            obs, reward, terminal, _ = game.step(action)
            obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
            obs = cv2.resize(obs, (128, 128))
            total_reward += reward
            fake_action = self.min_action(prev_obs, action, obs)

            D.append((prev_obs, action, fake_action))

            if len(D) >= args.batch_size:
                minibatch = random.sample(D, args.batch_size)
                obs_batch = [d[0] for d in minibatch]
                action_batch = [d[1] for d in minibatch]
                fake_action_batch = [d[2] for d in minibatch]

                _, loss_summary = sess.run([self.train_step, self.loss_summary], feed_dict={
                    self.state: obs_batch,
                    self.action: action_batch,
                    self.fake_action: fake_action_batch})

                if not self.experiment:
                    self.summary_writer.add_summary(loss_summary, t)

            print(t)
            if t % 500 == 0:
                print("Evaluating", t)
                self.eval_policy(game, t)
                terminal = True

if not os.path.exists(args.exp_dir):
    os.makedirs(args.exp_dir)

for exp in range(0, 100):
    print("Running experiment", exp)

    tf.reset_default_graph()
    exp_writer = open(args.exp_dir + "/" + str(exp) + ".csv", "w")
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    with sess.as_default():
        ilpo = Policy(sess=sess, shape=[None, 128, 128, 3], input_action=True, verbose=False, experiment=True, exp_writer=exp_writer)
        ilpo.run_policy()
        exp_writer.close()

