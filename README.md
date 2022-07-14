# FORLORN: Comparing Offline Methods and Reinforcement Learning for RAN Parameter Optimization

This code accompanies the following paper:

> Vegard Edvardsen, Gard Spreemann, and Jeriek Van den Abeele (2022). FORLORN: A Framework for Comparing Offline Methods and Reinforcement Learning for Optimization of RAN Parameters. _Submitted to the 25th ACM Intl. Conf. on Modeling, Analysis and Simulation of Wireless and Mobile Systems (MSWiM '22)_.

## Introduction

FORLORN enables a workflow for developing Reinforcement Learning (RL)-based optimization algorithms for Radio Access Networks (RAN). The framework builds on top of network scenarios implemented in the ns-3 network simulator, letting RAN parameters be optimized by RL agents implemented with the Stable-Baselines3 (SB3) library. Crucially, in order to understand and compare the agent's performance, FORLORN also integrates the black-box optimization framework Optuna to provide a _non-RL benchmark_. The framework allows easy comparison between RL (SB3) and non-RL (Optuna) performance through RL agent _scorecards_ produced across the various test scenarios. A more detailed description is provided in the accompanying paper.

## Code overview

The following files implement the C++ part of the code, i.e. the network simulation scenario on top of ns-3:

| File | Contents |
| --- | --- |
| `simulator.cc` | Network scenario main program. Text-based IPC with the Python code |
| `mobility.cc` | Implementation of our custom mobility model |
| `mobility.h` | Headers for our custom mobility model |
| `Makefile` | Makefile rules for compiling the simulator and linking to ns-3 |

The following Python code implements and integrates the various parts of the FORLORN framework:

| File | Contents |
| --- | --- |
| `simulator.py` | Interface between the network simulator and the Python code. Not RL-specific |
| `environment.py` | Implementation of RL environment (actions, observations, reward) |
| `trainer.py` | Script to train, save, load, and run the RL agent |
| `visualizer.py` | Script to visualize a simulation, based on its output log. Uses Gnuplot for rendering |
| `optimizer.py` | Script to use Optuna for offline black-box optimization |
| `hyperparams.py` | Script to use Optuna for hyperparameter tuning for the RL agent |

The following Jupyter notebooks support the development and training workflow enabled by FORLORN:

| File | Contents |
| --- | --- |
| `hyperparams.ipynb` | Inspecting and visualizing hyperparameter tuning results from `hyperparams.py` |
| `scorecard.ipynb` | Visualizing Optuna/RL agent performance, and generating RL agent's scorecard |
| `findscenarios.ipynb` | Helper notebook to sweep through RNG seeds in search for particular scenarios |

## Initial setup

The following commands should suffice to set up the necessary packages on a Debian 11 system (the current stable version as of July 2022, Debian 11 bullseye):

```
apt-get purge python3-torch
apt-mark hold python3-torch

apt-get install build-essential libns3-dev gnuplot-nox ffmpeg \
    ipython3 python3-numpy python3-scipy python3-matplotlib python3-pip

pip3 install protobuf==3.20.1
pip3 install torch==1.8.2+cpu torchvision==0.9.2+cpu torchaudio==0.8.2 \
    -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip3 install tensorboard==2.8.0 gym==0.19.0 stable-baselines3==1.2.0 optuna==2.10.0
```

Please note that while we have specified here the exact versions we used under Debian 11, this might need adaptation to your specific system (we are e.g. aware of package issues on ARM architecture). Depending on your specific circumstances, you might want to run the most recent package versions and/or those distributed with your OS.

Finally, to compile the simulator and set up necessary output folders:

```
make
mkdir output
```

The simulator was designed for ns-3 version 3.31. As of version 3.36, we are aware of changes to the `ns3::LteRrcSap::MeasResults` struct that may require minor code modifications.

## Typical workflows

These are some typical workflows in developing RL agents with FORLORN:

### 1. Finding interesting RNG seeds for the test scenarios

The simulated network setup in `simulator.cc` includes a _scenario generator_ that, given an RNG seed, samples UE locations in clusters across the environment (see `BaseNetworkScenario::define_ue_clusters` for how the seeds are mapped to UE locations). This allows the RL training process to generate an endless stream of similar—yet slightly different—situations within the same "family" of network scenarios. The range of RNG seeds used for RL training is defined in `default_train_levels` in `trainer.py`.

Likewise, the _test scenarios_ deemed salient enough to compare the agent's performance in the final scorecard, are also given as RNG seeds. For example, the test scenarios TS1–TS6 presented in the paper, correspond respectively to the RNG seeds `695 220 612 75 60 17`. These seeds are listed in the `scorecard.ipynb` notebook, where the RL agent scorecard is generated.

Sometimes it might be necessary to search for new specific scenarios to test the agent, e.g. if the scenario itself has changed or in order to test new aspects of the RL agent. In this case, the **[`findscenarios.ipynb`](./findscenarios.ipynb)** notebook shows how to easily loop through a range of RNG seeds in search for those that match a set of criteria.

Once you have arrived at a set of seeds you are satisfied with, make sure you use this new set of seeds when running the relevant commands listed below. The seeds should also be updated in `scorecard.ipynb`, as well as in `default_eval_levels` in `trainer.py`, which lists the scenarios used to guide Optuna during the RL agent's hyperparameter search.

### 2. Performing offline optimization of the test scenarios

To perform offline black-box optimization of the network scenarios using Optuna (in order to establish the benchmark for the RL agent), run the following commmand:

```
./optimizer.py optimize --grid --optuna 695 220 612 75 60 17
```

This will start 12 optimization "studies": for all of the six test scenarios listed on the command line, both a grid search study and a TPE study each. Each study will be launched in a separate parallel subprocess, running a series of 125 optimization trials per study. (Additional parallelism can be set using the `--parallelism` flag.)

The trial results (along with their complete output logs, for later trial visualization) will be saved in the SQLite database `scorecard.db`. (See workflow 5 below for how to visualize the results from this database.)

### 3. Hyperparameter tuning of the RL agent

Whenever the RL agent or the network scenario has changed sufficiently to warrant a new sweep of the hyperparameter space, this can be initiated as follows:

```
./hyperparams.py --n-trials 100 --train-timesteps 200000
```

This command will search through the hyperparameter space from scratch, sampling hyperparameter values from the ranges defined in `sample_algo_params()` and `sample_env_params()` in `hyperparams.py`. Alternatively, to only fine-tune a selection of the parameters (e.g. `learning_rate` and `step_size`) while using a default configuration file for the rest of them, use a command such as the following:

```
./hyperparams.py --n-trials 100 --train-timesteps 200000 --config config.json learning_rate step_size
```

The results from the hyperparameter search will be stored by Optuna in the SQLite database `hyperparams.db`. Use the notebook **[`hyperparams.ipynb`](./hyperparams.ipynb)** to inspect the results from the optimization session, and then update `config.json` with the chosen hyperparameter values.

### 4. Training the RL agent

Once you have a set of hyperparameters you are satisfied with, here's how to run a new, longer training session to get the final trained RL agent:

```
./trainer.py --tuned --train 1000000 --save model
```

The flag `--tuned` causes hyperparameters to be loaded from `config.json`. The trained agent (neural network weights etc.) will be serialized by SB3 into `model.zip`, while environment normalization statistics will be stored in `model.norm` (used by SB3's VecNormalize wrapper to normalize observations and rewards).

Training progress is logged in TensorBoard format to the folder `tblog` by default. You can follow the training progress using TensorBoard by running `tensorboard --logdir tblog` in a separate terminal.

### 5. Testing the RL agent across scenarios and generating its RL agent scorecard

With a trained RL agent (workflow 4) and with benchmark results from offline optimization (workflow 2) already established, it is time to compare the RL agent's performance in the test scenarios. First, loop through the test scenarios and run the RL agent for a number of test trials in each:

```
for seed in 695 220 612 75 60 17; do for n in $(seq 100); do
    ./trainer.py --tuned --load model --run-level $seed --run >output/log-$seed-$n.txt;
done; done
```

This will load the serialized RL agent from `model.zip`/`model.norm`, run the agent a number of times in each scenario and save the output logs to `output/log-NNN-MMM.txt`. Next, we import these output logs into the same SQLite database used by Optuna to track the grid search/TPE results:


```
./optimizer.py import output/log-*.txt
```

This command will import the RL trials into `scorecard.db` (under distinct Optuna studies, to separate them from the grid search/TPE results). These results can now be visualized using the **[`scorecard.ipynb`](./scorecard.ipynb)** notebook. The notebook will loop through each test scenario found in the SQLite database, visualize the simulation state for the best grid search/TPE/RL trial in each, and present these results in a table. Finally, the notebook will generate the full scorecard collating all of these results.

### 6. Running/visualizing a long-running trial with a sequence of scenarios

As we demonstrate in the paper, it is possible for the RL agent to continue to optimize the network parameters in a long-running trial where users move between their positions corresponding to different test scenarios. First, to run such an instance of a long-running trial, give a comma-separated list of seeds for the users to visit in sequence:

```
./trainer.py --tuned --load model --run-level 695,220,612,695,220,612,695 --run-duration 500000 --run >longtrial.txt
```

The course of this trial can be rendered into an animation by first exporting the benchmark values to a JSON file, then invoking the visualization script to generate a sequence of animation frames, before finally using FFMPEG to encode these into a video:

```
./optimizer.py results results.json
./visualizer.py --frame-skip 10 --results results.json longtrial.txt
ffmpeg -framerate 10 -i output/frame%d.png -pix_fmt yuv420p movie.mp4
```

## License

Copyright (c) 2022 Telenor ASA

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
