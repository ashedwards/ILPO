"""Base class for ILPO."""
"""Modified from https://github.com/affinelayer/pix2pix-tensorflow"""
from utils import *


class ILPO():
    def load_examples(self):
        """Loads in training examples."""

        raise NotImplementedError

    def create_encoder(self, state):
        """Creates an encoding of the state."""

        raise NotImplementedError

    def create_generator(self, layers, generator_output_channels):
        """Creates a generator for making next state predictions."""

        raise NotImplementedError

    def train_examples(self, examples):
        """Trains the model. TODO: Place method here."""

        raise NotImplementedError

    def process_inputs(self):
        """Processes the inputs used for the ILPO policy."""

        raise NotImplementedError

    def encode(self, state):
        """Runs an encoding on a state."""

        with tf.variable_scope("ilpo_loss", reuse=True):
            with tf.variable_scope("state_encoding"):
                return self.create_encoder(state)

    def create_ilpo(self, s_t, generator_outputs_channels):
        """Creates ILPO network."""

        # Create state embeddings.
        with tf.variable_scope("state_encoding"):
            s_t_layers = self.create_encoder(s_t)

        # Predict latent action probabilities.
        with tf.variable_scope("action"):
            flat_s = lrelu(s_t_layers[-1], .2)
            action_prediction = fully_connected(flat_s, args.n_actions)

            for a in range(args.n_actions):
                tf.summary.histogram("action_{}".format(a), action_prediction[:,a])
            tf.summary.histogram("action_max", tf.nn.softmax(action_prediction))

            action_prediction = tf.nn.softmax(action_prediction)

        # predict next state from latent action and current state.
        outputs = []
        shape = [ind for ind in flat_s.shape]
        shape[0] = tf.shape(flat_s)[0]

        for a in range(args.n_actions):
            # there is one generator g(s,z) that takes in a state s and latent action z.
            with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
                action = tf.one_hot([a], args.n_actions)

                # obtain fully connected latent action to concatenate with state.
                action = fully_connected(action, int(flat_s.shape[-1]), reuse=tf.AUTO_REUSE, scope="action_embedding")
                action = lrelu(action, .2)

                # tile latent action embedding.
                action = tf.tile(action, [1, tf.shape(s_t)[0]])
                action = tf.reshape(action, shape)

                # concatenate state and action.
                state_action = slim.flatten(tf.concat([flat_s, action], axis=-1))
                state_action = fully_connected(state_action, int(flat_s.shape[-1]), reuse=tf.AUTO_REUSE, scope="state_action_embedding")
                state_action = tf.reshape(state_action, tf.shape(flat_s))

                s_t_layers[-1] = state_action
                outputs.append(self.create_generator(s_t_layers, generator_outputs_channels))

        expected_states = 0
        shape = tf.shape(outputs[0])

        # compute expected next state as sum_z p(z|s)*g(s,z)
        for a in range(args.n_actions):
            expected_states += tf.multiply(tf.stop_gradient(slim.flatten(outputs[a])), tf.expand_dims(action_prediction[:, a], -1))
        expected_states = tf.reshape(expected_states, shape)

        return (expected_states, outputs, action_prediction)

    def create_model(self, inputs, targets):
        """ Initializes ILPO model and losses."""

        global_step = tf.contrib.framework.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

        with tf.variable_scope("ilpo_loss"):
            out_channels = int(targets.get_shape()[-1])
            expected_outputs, outputs, actions = self.create_ilpo(inputs, out_channels)

            # compute loss on expected next state.
            delta = slim.flatten(targets - inputs)
            gen_loss_exp = tf.reduce_mean(
                tf.reduce_sum(tf.losses.mean_squared_error(delta, slim.flatten(expected_outputs),
                                                   reduction=tf.losses.Reduction.NONE), axis=1))

            # compute loss on min next state.
            all_loss = []

            for out in outputs:
                all_loss.append(tf.reduce_sum(
                    tf.losses.mean_squared_error(delta, slim.flatten(out),
                    reduction=tf.losses.Reduction.NONE),
                    axis=1))

            stacked_min_loss = tf.stack(all_loss, axis=-1)
            gen_loss_min = tf.reduce_mean(tf.reduce_min(stacked_min_loss, axis=1))

            gen_loss_L1 = gen_loss_exp + gen_loss_min

            # obtain images and scalars for summaries.
            tf.summary.scalar("expected_gen_loss", gen_loss_exp)
            tf.summary.scalar("min_gen_loss", gen_loss_min)

            min_index = tf.argmin(all_loss)
            min_index = tf.one_hot(min_index, args.n_actions)

            shape = tf.shape(out)
            min_img = tf.stack([slim.flatten(out) for out in outputs], axis=-1)
            min_img = tf.reduce_sum(tf.multiply(min_img, tf.expand_dims(min_index, 1)), -1)
            min_img = tf.reshape(min_img, shape)


        with tf.name_scope("ilpo_train"):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("ilpo")]
            gen_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss_L1, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        return Model(
            gen_loss_L1=gen_loss_L1,
            gen_grads_and_vars=gen_grads_and_vars,
            outputs=[inputs + out for out in outputs],
            expectation=inputs + expected_outputs,
            min_output=inputs + min_img,
            actions=actions,
            train=tf.group(gen_loss_L1, incr_global_step, gen_train),
        )


    def run(self):
        """Runs training method."""

        if tf.__version__.split('.')[0] != "1":
            raise Exception("Tensorflow version 1 required")

        if args.seed is None:
            args.seed = random.randint(0, 2**31 - 1)

        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        if args.mode == "test" or args.mode == "export":
            if args.checkpoint is None:
                raise Exception("checkpoint required for test mode")

            # load some options from the checkpoint
            options = {"which_direction", "ngf", "ndf", "lab_colorization"}
            with open(os.path.join(args.checkpoint, "options.json")) as f:
                for key, val in json.loads(f.read()).items():
                    if key in options:
                        print("loaded", key, "=", val)
                        setattr(a, key, val)
            # disable these features in test mode
            args.flip = False

        for k, v in args._get_kwargs():
            print(k, "=", v)

        with open(os.path.join(args.output_dir, "options.json"), "w") as f:
            f.write(json.dumps(vars(args), sort_keys=True, indent=4))

        examples = self.load_examples()
        self.train_examples(examples)
