# Imitating Latent Policies from Observation (ILPO) [[Paper]](https://arxiv.org/abs/1805.07914)
Ashley D. Edwards, Himanshu Sahni, Yannick Schroecker, Charles L. Isbell</br>
Georgia Institute of Technology

<img src="https://github.gatech.edu/ashedwards/ILPO/blob/master/resources/network.png" width="600">

## Abstract
We describe a novel approach to imitation learning that infers latent policies directly from state observations. We introduce a method that characterizes the causal effects of unknown actions on observations while simultaneously predicting their likelihood. We then outline an action alignment procedure that leverages a small amount of environment interactions to determine a mapping between latent and real-world actions. We show that this corrected labeling can be used for imitating the observed behavior, even though no expert actions are given. We evaluate our approach within classic control and photo-realistic visual environments and demonstrate that it performs well when compared to standard approaches.

If you use any of the code here in your own work, you may cite:

    @article{edwards2018imitating,
      title={Imitating Latent Policies from Observation},
      author={Edwards, Ashley D and Sahni, Himanshu and Schroecker, Yannick and Isbell, Charles L},
      journal={arXiv preprint arXiv:1805.07914},
      year={2018}
    }
    
## Getting started
This is the official implementation of the work [Imitating Latent Policies from Observation](https://arxiv.org/abs/1805.07914). This approach aims to learn policies directly from state observations by utilizing two key components 1) a Latent Policy Network (LPN) that utilizes a multimodal forward dynamics network to learn priors over latent actions and 2) an Action Remapping Network (ARN) that leverages environment interactions to map the latent actions to real ones, as summarized in Figure 1. 

Note: This is research code and we not currently plan to maintain it. 

## Requirements 
This implementation has been tested with Python 3.5 on OS X High Sierra and Ubuntu 14.04. 

```Shell
# 1) Clone repository 
git clone https://github.gatech.edu/ashedwards/ILPO.git
cd action_observation_network 

# 2) Install requirements
pip install -r requirements.txt
```

If you have trouble installing baselines on OS X, try running the following commands: 

```Shell
brew install mpich
pip install mpi4py
```

We made a [custom environment](https://github.gatech.edu/ashedwards/ILPO/tree/master/gym-thor) based on the [AI2-Thor](https://ai2thor.allenai.org/) platflorm. You need to install it if you plan to use it for your own experiments:

```Shell
pip install -e environments/gym-thor
```

## Collecting expert data 
ILPO uses expert state observations to learn latent policies. We used [OpenAI Baselines](https://github.com/openai/baselines) to obtain these trajectories. Collecting the data consists of two steps: 1) training the expert and 2) running the learned policy and saving the observed state trajectories to disk. Let's walk through how to collect data for cartpole.

```Shell
# 1) Train expert
python train_expert/train_cartpole.py

# 2) Collect state trajectories 
python train_expert/enjoy_cartpole.py
```

Once done running, the expert policy from step 1 is written to [final_models/cartpole.pkl](https://github.gatech.edu/aedwards8/action_observation_network/tree/master/final_models/cartpole). Then, step 2 loads and runs the policy and saves the observed states to [final_models/acrobot/cartpole.txt](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/final_models/cartpole/cartpole.txt). 

The code for collecting data can be found in [train_expert](https://github.gatech.edu/aedwards8/action_observation_network/tree/master/train_expert).

All data for cartpole and acrobot is already saved in [final_models](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/final_models/cartpole/cartpole.txt). The files for AI2-Thor were too big to include here, so you will need to follow the above steps to obtain the expert data.

## Training ILPO
After collecting the expert state observations, you can then train ILPO. Check out the [scripts](https://github.gatech.edu/aedwards8/action_observation_network/tree/master/scripts) directory to view all of the training scripts. Let's again see how to run cartpole: 

```Shell
# 1) Train latent policy network
./scripts/run_vector_ilpo_cartpole.sh

# 2) Train action remapping network
./scripts/run_vector_policy_cartpole.sh
```

The first step learns the latent policy network. You'll notice a few necessary arguments in the [script](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/scripts/run_vector_ilpo_cartpole.sh):

    --mode whether training or testing. This should always be train for training LPN
    --input_dir location of state trajectories 
    --n_actions number of latent actions being learned
    --batch_size the batch size
    --output_dir where model and checkpoints are saved 
    --max_epochs max training epochs 
    
You can view the results in "output_dir", in this case, "cartpole_ilpo", in tensorboard:

```Shell
# 1) View results
tensorboard --logdir cartpole_ilpo
```

The second step learns the action remapping network after loading in the LPN model from "output_dir". Let's look at the arguments from the training [script](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/scripts/run_vector_policy_acrobot.sh):

    --mode whether training or testing. This should always be test for training ARN
    --n_actions number of latent actions being learned
    --real_actions number of real actions 
    --batch_size the batch size
    --checkpoint where the LPN model is saved 
    --env the agent's environment
    --exp_dir where to save experiments
    --max_epochs max training epochs 
    --n_dims the observation space of the agent

We differentiate latent from real actions in case we want to learn more latent causes than actions. This would allow the network to learn difficult transitions such as bumping into walls. 

The experiments are saved into the directory specified in "exp_dir". Each experiment is saved as a .csv file. You can plot these results using the [results/plot_results.py](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/results/plot_results.py) script:

```Shell
# 1) Plot results
python results/plot_results.py
```

Note that this expects 100 experiments to have been saved as csv files, and this is currently hard-coded in. You'll also need to modify the file names if you run something other than cartpole. 

If you are using a virtual_env, you may need to install matplotlib with conda:
```Shell
conda install matplotlib
```

## Running Behavioral Cloning
We compared against behavioral cloning in or work. Here are the steps for cartpole:

```Shell
# 1) Train latent policy network
./scripts/run_vector_bc_cartpole.sh

# 2) Evaluate learned policy
./scripts/run_vector_policy_cartpole_bc.sh
```

## Running your own data and architectures
We have two different data representations in this code, one for states that are represented through vectors (like cartpole), which can be found in [models/vector_ilpo.py](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/vector_ilpo.py), and one for images (like AI2-Thor), found in [models/image_ilpo.py](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/image_ilpo.py). 

### Using your own vector data 
The vector representation expects trajectories to be in a text file of the form:

    [state] [next_state]
            
Each line in the file represents an observation. This demonstration must be in a folder that consists only of demonstrations of this form, as all of the contents of the directory will be parsed. See [final_models](https://github.gatech.edu/aedwards8/action_observation_network/tree/master/final_models) for examples. 

Once done, you can just copy one of the [scripts](https://github.gatech.edu/aedwards8/action_observation_network/tree/master/scripts) and modify it to use your own directory and arguments. 

### Using your own image data
The image representation expects trajectories to be in an image file of the form:

    [[A][B]]
        
where [A] is an image of the state and [B] is an image of the next state. These images are concatenated together side by side to form a single image. This demonstration must also be in a folder that consists only of demonstrations of this form. We do not have any examples for this, but it will automatically be created if you create expert data for AI2-Thor. 

This representation was borrowed from the pix2pix [implementation](https://github.com/affinelayer/pix2pix-tensorflow). So you could also try out the datasets from that code. While those are not RL demonstrations, LPN could potentially be used for learning multimodal outputs from static datasets. 

### Using different data representations 
You may also use your own data representation. In this case, you will need to modify the [load_examples](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/ilpo.py#L7) method in [models/vector_ilpo.py](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/vector_ilpo.py) or [models/image_ilpo.py](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/image_ilpo.py), or create your own class entirely. 

### Creating your own architecture 
ILPO is not tied to one single architecture. Cartpole, for example, uses fully-connected layer to define the LPN, while Thor uses convnets. You can use the ones already defined for vectors or images in [models/vector_ilpo.py](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/vector_ilpo.py) and [models/image_ilpo.py](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/image_ilpo.py), respectively, or you can create your own by inheriting from models/ilpo.py. You will need to implement the following methods:

[create_encoder](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/ilpo.py#L12) creates an encoding of the state

[create_generator](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/ilpo.py#L17) creates a generator for making next state predictions

[train_examples](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/ilpo.py#L22) trains the model 

[process_inputs](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/ilpo.py#L27) processes the inputs used for the ILPO policy

You will also need to create custom scripts for running the policies. You can simply clone [models/vector_policy.py](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/vector_policy.py) or [models/image_policy.py](https://github.gatech.edu/aedwards8/action_observation_network/blob/master/models/image_policy.py) if you're using images, and modify the Policy class to inherit from your custom class. 


