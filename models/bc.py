"""Base class for Behavioral Cloning"""
"""Modified from https://github.com/affinelayer/pix2pix-tensorflow"""
from utils import *


class BC():
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
        """Processes the inputs used for the BC policy."""

        raise NotImplementedError

    def create_bc(self, s_t, generator_outputs_channels):
        """Creates the Behavioral Cloning Network."""

        # Create state embeddings.
        with tf.name_scope("state_encoding"):
            s_t_layers = self.create_encoder(s_t)

        # Predict action.
        with tf.name_scope("action"):
            flat_s = lrelu(s_t_layers[-1], .2)
            action_prediction = fully_connected(flat_s, args.n_actions)

            for a in range(args.n_actions):
                tf.summary.histogram("action_{}".format(a), action_prediction[:,a])
            tf.summary.histogram("action_max", tf.nn.softmax(action_prediction))

        return (action_prediction)

    def create_model(self, inputs, actions):
        """ Initializes BC model and losses."""

        global_step = tf.contrib.framework.get_or_create_global_step()
        incr_global_step = tf.assign(global_step, global_step+1)

        with tf.variable_scope("bc_action"):
            out_channels = int(inputs.get_shape()[-1])
            predicted_actions = self.create_bc(inputs, out_channels)

        with tf.name_scope("bc_loss"):
            bc_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=actions, logits=predicted_actions))
            tf.summary.scalar("bc_loss", bc_loss)

        with tf.name_scope("bc_train"):
            gen_tvars = [var for var in tf.trainable_variables()]
            gen_optim = tf.train.AdamOptimizer(args.lr, args.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(bc_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

        return BCModel(
            gen_loss_L1=bc_loss,
            gen_grads_and_vars=gen_grads_and_vars,
            actions=tf.nn.softmax(predicted_actions),
            train=tf.group(bc_loss, incr_global_step, gen_train),
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
