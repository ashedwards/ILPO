"""BC network for vectors."""
from bc import BC
from utils import *

class VectorBC(BC):
    def process_inputs(self, inputs):
        return inputs

    def load_examples(self):
        if args.input_dir is None or not os.path.exists(args.input_dir):
            raise Exception("input_dir does not exist")

        input_paths = glob.glob(os.path.join(args.input_dir, "*.txt"))

        if len(input_paths) == 0:
            raise Exception("input_dir contains no demonstration files")

        def get_name(path):
            name, _ = os.path.splitext(os.path.basename(path))
            return name

        # if the txt names are numbers, sort by the value rather than asciibetically
        # having sorted inputs means that the outputs are sorted in test mode
        if all(get_name(path).isdigit() for path in input_paths):
            input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
        else:
            input_paths = sorted(input_paths)

        with tf.name_scope("load_demonstrations"):
            paths = []
            inputs = []
            targets = []

            # Demonstrations should be of the form [s,s'] on each line in the file.
            for demonstration in input_paths:
                for trajectory in open(demonstration):
                    s, action, s_prime = trajectory.replace("\n", "").split(" ")
                    s = eval(s)
                    action = eval(action)
                    inputs.append(s)
                    targets.append(action)
                    paths.append(trajectory)

        num_samples = len(inputs)

        inputs = tf.convert_to_tensor(inputs, tf.float32)
        targets = tf.one_hot(tf.squeeze(tf.convert_to_tensor(targets, tf.int32)), args.n_actions)

        paths_batch, inputs_batch, targets_batch = tf.train.shuffle_batch(
            [paths, inputs, targets],
            batch_size=args.batch_size,
            num_threads=1,
            enqueue_many=True,
            capacity=num_samples,
            min_after_dequeue=1000)

        inputs_batch.set_shape([args.batch_size, inputs_batch.shape[-1]])
        targets_batch.set_shape([args.batch_size, targets_batch.shape[-1]])
        steps_per_epoch = int(math.ceil(num_samples / args.batch_size))

        return Examples(
            paths=paths_batch,
            inputs=inputs_batch,
            targets=targets_batch,
            count=len(input_paths),
            steps_per_epoch=steps_per_epoch,
        )


    def create_encoder(self, state):
        """Creates state embedding."""

        layers = []

        # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
        with tf.variable_scope("encoder_1"):
            output = fully_connected(state, args.ngf)
            layers.append(output)

        layer_specs = [
            args.ngf * 2,
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                encoded = fully_connected(rectified, out_channels)
                layers.append(encoded)

        return layers

    def create_generator(self, layers, generator_outputs_channels):
        s_t_layers = list(layers)

        with tf.variable_scope("decoder_1"):
            inp = s_t_layers[-1]
            rectified = lrelu(inp, 0.2)
            output = fully_connected(rectified, args.ngf)
            s_t_layers.append(output)

        with tf.variable_scope("decoder_2"):
            inp = s_t_layers[-1]
            rectified = lrelu(inp, 0.2)
            output = fully_connected(rectified, generator_outputs_channels)
            s_t_layers.append(output)

        return s_t_layers[-1]

    def train_examples(self, examples):
        print("examples count = %d" % examples.count)

        # inputs and targets are [batch_size, height, width, channels]
        model = self.create_model(examples.inputs, examples.targets)
        inputs = examples.inputs
        targets = examples.targets

        tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name + "/values", var)

        for grad, var in model.gen_grads_and_vars:
            tf.summary.histogram(var.op.name + "/gradients", grad)

        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

        saver = tf.train.Saver(max_to_keep=1)

        logdir = args.output_dir if (args.trace_freq > 0 or args.summary_freq > 0) else None
        sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.20)

        with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            print("parameter_count =", sess.run(parameter_count))

            if args.checkpoint is not None:
                print("loading model from checkpoint")
                checkpoint = tf.train.latest_checkpoint(args.checkpoint)
                saver.restore(sess, checkpoint)

            max_steps = 2 ** 32
            if args.max_epochs is not None:
                max_steps = examples.steps_per_epoch * args.max_epochs
            if args.max_steps is not None:
                max_steps = args.max_steps

            if args.mode == "test":
                # testing
                # at most, process the test data once
               pass
            else:
                # training
                start = time.time()

                for step in range(max_steps):
                    def should(freq):
                        return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

                    options = None
                    run_metadata = None
                    if should(args.trace_freq):
                        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                    fetches = {
                        "train": model.train,
                        "global_step": sv.global_step,
                    }

                    if should(args.progress_freq):
                        fetches["gen_loss_L1"] = model.gen_loss_L1


                    if should(args.summary_freq):
                        fetches["summary"] = sv.summary_op

                    if should(args.display_freq):
                        fetches["display"] = display_fetches

                    results = sess.run(fetches, options=options, run_metadata=run_metadata)

                    if should(args.summary_freq):
                        print("recording summary")
                        sv.summary_writer.add_summary(results["summary"], results["global_step"])

                    if should(args.trace_freq):
                        print("recording trace")
                        sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

                    if should(args.progress_freq):
                        # global_step will have the correct step count if we resume from a checkpoint
                        train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                        train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                        rate = (step + 1) * args.batch_size / (time.time() - start)
                        remaining = (max_steps - step) * args.batch_size / rate
                        print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                        train_epoch, train_step, rate, remaining / 60))
                        print("gen_loss_L1", results["gen_loss_L1"])

                    if should(args.save_freq):
                        print("saving model")
                        saver.save(sess, os.path.join(args.output_dir, "model"), global_step=sv.global_step)

                    if sv.should_stop():
                        break
def main():
    model = VectorBC()
    model.run()

if __name__ == "__main__":
    main()
