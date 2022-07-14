#!/usr/bin/env python3
#
# FORLORN - Comparing Offline Methods and RL for RAN Parameter Optimization
# Copyright (c) 2022 Telenor ASA
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
# -----
#
# This code accompanies the following paper: Vegard Edvardsen, Gard Spreemann,
# and Jeriek Van den Abeele. 2022. FORLORN: A Framework for Comparing Offline
# Methods and Reinforcement Learning for Optimization of RAN Parameters.
# Submitted to the 25th ACM International Conference on Modeling, Analysis and
# Simulation of Wireless and Mobile Systems (MSWiM '22).
#
# -----
#
# hyperparams.py - Hyperparameter tuning for RL
#
# Command line script to tune the hyperparameters of the RL agent using Optuna.
# Parameters either related to the RL model itself (A2C or PPO), the training
# process or our own parameters for the environment can all be tuned

import argparse
import copy
import json
import optuna
import stable_baselines3 as sb3

import trainer

class HyperparameterTuner:
    def __init__(self, base_config, params_to_tune, train_timesteps=24000, mock=False):
        self.base_config = base_config
        self.params_to_tune = params_to_tune
        self.train_timesteps = train_timesteps
        self.mock = mock

        # Either need to provide a base config, or fine-tune all the parameters
        assert len(self.base_config) > 0 or len(self.params_to_tune) == 0

        self.study = optuna.create_study(
            storage='sqlite:///hyperparams.db', study_name='rl', load_if_exists=True,
            sampler=optuna.samplers.TPESampler(), direction='maximize')

        self.trial = None
        self.config = None

    def run(self, n_trials):
        self.study.optimize(self.perform_trial, n_trials=n_trials)

    def sample_param(self, param, param_type, *param_args):
        if len(self.params_to_tune) > 0 and param not in self.params_to_tune:
            return
        if param_type == 'categorical':
            self.config[param] = self.trial.suggest_categorical(param, *param_args)
        elif param_type == 'uniform':
            self.config[param] = self.trial.suggest_uniform(param, *param_args)
        elif param_type == 'loguniform':
            self.config[param] = self.trial.suggest_loguniform(param, *param_args)
        elif param_type == 'int':
            self.config[param] = self.trial.suggest_int(param, *param_args)

    def sample_algo_params(self):
        # Sample which RL algorithm to use (A2C or PPO)
        self.sample_param('algo', 'categorical', ['a2c', 'ppo'])

        # Sample hyperparameters common to A2C and PPO (following parts adapted from
        # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py)
        self.sample_param('ent_coef', 'loguniform', 0.00000001, 0.1)
        self.sample_param('gae_lambda', 'categorical', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
        self.sample_param('gamma', 'categorical', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
        self.sample_param('learning_rate', 'loguniform', 1e-5, 1)
        self.sample_param('max_grad_norm', 'categorical', [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
        self.sample_param('n_steps', 'categorical', [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        self.sample_param('vf_coef', 'uniform', 0, 1)

        # Hyperparameters that are specific to A2C or PPO
        if self.config['algo'] == 'a2c':
            self.sample_param('normalize_advantage', 'categorical', [False, True])
            self.sample_param('use_rms_prop', 'categorical', [False, True])
        if self.config['algo'] == 'ppo':
            self.sample_param('batch_size', 'categorical', [8, 16, 32, 64, 128, 256, 512])
            self.sample_param('clip_range', 'categorical', [0.1, 0.2, 0.3, 0.4])
            self.sample_param('n_epochs', 'categorical', [1, 5, 10, 20])

        # Hyperparameters for the policy network (see trainer.extract_algo_args)
        self.sample_param('activation_fn', 'categorical', ['tanh', 'relu'])
        self.sample_param('net_arch', 'categorical', ['small', 'medium'])
        self.sample_param('ortho_init', 'categorical', [False, True])

    def sample_env_params(self):
        # Set various default config that is used by trainer.get_train_env and
        # get_eval_env when constructing the training and eval envs below
        self.config['mock'] = self.mock
        self.config['quiet'] = True
        self.config['ue_count'] = 12
        self.config['eval_duration'] = 10000
        self.config['train_levels'] = trainer.default_train_levels
        self.config['eval_levels'] = trainer.default_eval_levels
        self.config['rsrq_quantile_fractions'] = None

        # Sample the environment-specific hyperparameters we wish to tune
        self.sample_param('n_envs', 'categorical', [4, 8, 16])
        self.sample_param('train_duration', 'categorical', [5000, 10000, 15000, 20000])
        self.sample_param('randomize', 'categorical', [False, True])
        self.sample_param('history', 'int', 1, 20)
        self.sample_param('step_size', 'int', 1, 10)
        self.sample_param('num_rsrq_quantiles', 'int', 1, 5)
        self.sample_param('oob_penalty_factor', 'categorical', [1e-4, 1e-2, 1.0])
        self.sample_param('oob_means_gameover', 'categorical', [False, True])

    def perform_trial(self, trial):
        # Start with base config and sample new values for the parameters to tune
        self.trial = trial
        self.config = copy.deepcopy(self.base_config)
        self.sample_algo_params()
        self.sample_env_params()

        # Construct training and evaluation environments
        train_env = trainer.get_train_env(self.config)
        eval_env = trainer.get_eval_env(self.config)

        # Contruct and train the RL model
        model = trainer.create_rl_model(self.config, train_env, verbose=0)
        last_eval_score = trainer.train_model(self.config, model, eval_env, self.train_timesteps)

        # Close the training/eval envs before returning with the score
        train_env.close()
        eval_env.close()
        return last_eval_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=100, help='Number of trials to run')
    parser.add_argument('--train-timesteps', type=int, default=24000, help='Number of timesteps for each trial')
    parser.add_argument('--mock', action='store_true', help='Use the mock version of the simulator')
    parser.add_argument('--config', help='Base config for default parameter values')
    parser.add_argument('params', nargs='*', help='List of params to fine-tune (defaults to all if omitted)')
    args = parser.parse_args()

    base_config = {}
    if args.config is not None:
        with open(args.config, 'r') as f:
            base_config = json.load(f)

    HyperparameterTuner(base_config, args.params, args.train_timesteps, args.mock).run(args.n_trials)
