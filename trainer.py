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
# trainer.py - Training script for Reinforcement Learning
#
# Command line script to run different Reinforcement Learning training
# algorithms from Stable Baselines 3 on our network simulation environment.
# Trained models can be serialized to disk and later reloaded, e.g. to continue
# training or to evaluate the behavior of the trained agent in new trials

import argparse
import json
import stable_baselines3 as sb3
import sys
import torch

import environment

def parse_range(text):
    # Parse a textual range representation such as 1-100 (inclusive on both ends)
    if ';' in text:
        return sum(map(parse_range, text.split(';')), start=[])
    if '-' in text:
        ends = list(map(int, text.split('-')))
        return list(map(str, range(ends[0], ends[1] + 1)))
    else:
        return [ text ]

def parse_float_list(text):
    # Parse a comma-separated list of floats
    return list(map(float, text.split(',')))

# Specify our default sets of training/eval levels here, so that these are
# accessible to other modules such as hyperparams.py. Default eval levels were
# found by searching for interesting seeds (see findscenarios.ipynb)
default_train_levels = parse_range('1000-1999')
default_eval_levels = parse_range('434;695;298;612;220;254;75;104;60;80;15;17')

def get_env_init_fn(config, duration, randomize, env_levels, oob_penalty_factor, oob_means_gameover):
    return lambda: sb3.common.monitor.Monitor(
        environment.Environment(
            # Parameters to the environment
            duration=duration,
            history=config['history'],
            step_size=config['step_size'],
            num_rsrq_quantiles=config['num_rsrq_quantiles'],
            rsrq_quantile_fractions=config['rsrq_quantile_fractions'],
            oob_penalty_factor=oob_penalty_factor,
            oob_means_gameover=oob_means_gameover,

            # Parameters to the simulator
            levels=env_levels,
            ue_count=config['ue_count'],
            mock=config['mock'],
            randomize=randomize,
            quiet=config['quiet'],
    ))

def create_vec_env_with_levels(config, duration, randomize, levels_per_env, oob_penalty_factor, oob_means_gameover):
    return sb3.common.vec_env.SubprocVecEnv([
        get_env_init_fn(config, duration, randomize, env_levels, oob_penalty_factor, oob_means_gameover)
        for env_levels in levels_per_env
    ], start_method='fork')

def wrap_with_vec_normalize(vec_env, model_load_filename, training):
    '''
    Wrap the vectorized environment in a normalization wrapper, either loading
    a previous run from file or starting from scratch. Depending on whether
    we're wrapping the training, evaluation or run env, we might want to turn
    on/off updating the normalization statistics and normalization of rewards
    '''
    if model_load_filename:
        print('Loading normalization statistics', file=sys.stderr)
        filename = '%s.norm' % model_load_filename
        vec_norm = sb3.common.vec_env.VecNormalize.load(filename, vec_env)
    else:
        print('Initializing empty normalization statistics', file=sys.stderr)
        vec_norm = sb3.common.vec_env.VecNormalize(vec_env)
    vec_norm.training = training
    vec_norm.norm_reward = training
    return vec_norm

def get_train_env(config, model_load_filename=None, full_training_env=True):
    if not full_training_env:
        return get_run_env(config)
    vec_env = create_vec_env_with_levels(config,
        duration=config['train_duration'], randomize=config['randomize'],
        levels_per_env=[ config['train_levels'] for i in range(config['n_envs']) ],
        oob_penalty_factor=config['oob_penalty_factor'],
        oob_means_gameover=config['oob_means_gameover'])
    return wrap_with_vec_normalize(vec_env, model_load_filename, training=True)

def get_eval_env(config, model_load_filename=None):
    vec_env = create_vec_env_with_levels(config,
        duration=config['eval_duration'], randomize=False,
        levels_per_env=[ [eval_level] for eval_level in config['eval_levels'] ],
        oob_penalty_factor=0.0,
        oob_means_gameover=False)
    return wrap_with_vec_normalize(vec_env, model_load_filename, training=False)

def get_run_env(config, model_load_filename=None):
    vec_env = create_vec_env_with_levels(config,
        duration=config['run_duration'], randomize=True,
        levels_per_env=[[config['run_level']]],
        oob_penalty_factor=0.0,
        oob_means_gameover=False)
    return wrap_with_vec_normalize(vec_env, model_load_filename, training=False)

def extract_algo_args(config):
    # Determine which parameters to extract from the config (some are common to
    # both A2C and PPO, while other are A2C/PPO specific)
    keys_to_extract = [ 'ent_coef', 'gae_lambda', 'gamma', 'learning_rate',
        'max_grad_norm', 'n_steps', 'vf_coef' ]
    if config['algo'] == 'a2c':
        keys_to_extract += [ 'normalize_advantage', 'use_rms_prop' ]
    if config['algo'] == 'ppo':
        keys_to_extract += [ 'batch_size', 'clip_range', 'n_epochs' ]

    # Extract the parameters, and adjust batch size if too large
    algo_args = { key: config[key] for key in keys_to_extract if key in config }
    if 'batch_size' in config and config['batch_size'] > config['n_steps'] * config['n_envs']:
        algo_args['batch_size'] = config['n_steps'] * config['n_envs']

    # Extract parameters to the policy network (MlpPolicy)
    if 'activation_fn' in config:
        algo_args['policy_kwargs'] = {
            'activation_fn': {
                'relu': torch.nn.ReLU,
                'tanh': torch.nn.Tanh,
            }[config['activation_fn']],
            'net_arch': {
                'small': [dict(pi=[64, 64], vf=[64, 64])],
                'medium': [dict(pi=[256, 256], vf=[256, 256])],
            }[config['net_arch']],
            'ortho_init': config['ortho_init'],
        }
    return algo_args

rl_algo_classes = {
    'a2c': sb3.A2C,
    'ppo': sb3.PPO,
}

def create_rl_model(config, train_env, tensorboard=True, verbose=1):
    algo_class = rl_algo_classes[config['algo']]
    algo_args = extract_algo_args(config)
    return algo_class(
        policy='MlpPolicy', env=train_env, verbose=verbose,
        tensorboard_log='./tblog/' if tensorboard else None,
        **algo_args)

def load_rl_model(config, train_env, model_load_filename):
    algo_class = rl_algo_classes[config['algo']]
    return algo_class.load(model_load_filename, env=train_env)

def train_model(config, model, eval_env, timesteps):
    eval_callback = sb3.common.callbacks.EvalCallback(eval_env,
        n_eval_episodes=len(config['eval_levels']),
        eval_freq=(4000 // config['n_envs']))
    try:
        model.learn(total_timesteps=timesteps, log_interval=1, callback=eval_callback)
    except BaseException as e:
        print(e)
    return eval_callback.last_mean_reward

if __name__ == '__main__':
    # Arguments to control the overall operation of the script
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', metavar='FILE', help='Load the model from FILE.zip')
    parser.add_argument('--save', metavar='FILE', help='Save the model to FILE.zip')
    parser.add_argument('--train', type=int, default=0, metavar='N', help='Train the model for N timesteps')
    parser.add_argument('--run', action='store_true', help='Run the model for a single trial')
    parser.add_argument('--tensorboard', action=argparse.BooleanOptionalAction, default=True, help='Use Tensorboard logging')

    # Arguments related to the RL algorithm
    parser.add_argument('--algo', choices=['a2c', 'ppo'], default='a2c', help='Which RL algorithm to use')
    parser.add_argument('--tuned', action='store_true', help='Use the fine-tuned version of the agent')

    # Arguments related to the environment
    parser.add_argument('--n-envs', type=int, default=8, help='Number of parallel environments during training')
    parser.add_argument('--train-duration', type=int, default=10000, help='Duration of a training episode (in ms)')
    parser.add_argument('--eval-duration', type=int, default=10000, help='Duration of an evaluation episode (in ms)')
    parser.add_argument('--run-duration', type=int, default=30000, help='Duration of a single-trial run (in ms)')
    parser.add_argument('--history', type=int, default=1, help='Number of historical steps included in agent observation')
    parser.add_argument('--step-size', type=int, default=10, help='Tenths of dBm the agent can change transmission power per timestep')
    parser.add_argument('--num-rsrq-quantiles', type=int, help='Use this many RSRQ quantiles as observations (for each eNB); equivalent to --rsrq-quantile-fractions 1/n,2/n,...,(n-1)/n')
    parser.add_argument('--rsrq-quantile-fractions', type=parse_float_list, help='Use these fractions (in [0,1]) for RSRQ quantiles as observations (for each eNB)')
    parser.add_argument('--oob-penalty-factor', type=float, default=0.01, help='Penalize by factor*(number of UEs in simulation) if agent attempts to set power OOB _during training_ (no effect otherwise)')
    parser.add_argument('--oob-means-gameover', action='store_true', help='Power settings out of bounds _during training_ is considered a game over; setting this also means that --oob-penalty-factor is ignored; no effect when not training')

    # Arguments related to the simulator
    parser.add_argument('--train-levels', type=parse_range, default=default_train_levels, help='Levels to use during training')
    parser.add_argument('--eval-levels', type=parse_range, default=default_eval_levels, help='Levels to use during evaluation')
    parser.add_argument('--run-level', default='1', help='Level to use for single-trial runs')
    parser.add_argument('--ue-count', type=int, default=12, help='Total UE count')
    parser.add_argument('--mock', action='store_true', help='Use the mock version of the simulator')
    parser.add_argument('--randomize', action='store_true', help='Randomize initial base station settings')
    parser.add_argument('--quiet', action='store_true', help='Produce minimal output from the simulator')

    args = parser.parse_args()
    args.load = args.load[:-4] if args.load and args.load.endswith('.zip') else args.load
    args.save = args.save[:-4] if args.save and args.save.endswith('.zip') else args.save

    # Extract command line arguments into a config dict. The config can then be
    # overridden by fine-tuned hyperparameters loaded from config.json (as
    # returned from Optuna in study.best_params; see hyperparams.ipynb notebook)
    config = vars(args)
    if args.tuned:
        with open('config.json', 'r') as f:
            tuned_config = json.load(f)
            config.update(tuned_config)

    print('Initializing environment', file=sys.stderr)
    train_env = get_train_env(config, args.load, args.train > 0)
    eval_env = get_eval_env(config, args.load) if args.train > 0 else None
    run_env = get_run_env(config, args.load) if args.run_level is not None else None

    if args.load:
        print('Loading model from file', file=sys.stderr)
        model = load_rl_model(config, train_env, model_load_filename=args.load)
    else:
        print('Creating a new model', file=sys.stderr)
        model = create_rl_model(config, train_env, tensorboard=args.tensorboard)

    if args.train > 0:
        print('Training for up to %d timesteps (Ctrl-C to interrupt)' % args.train, file=sys.stderr)
        train_model(config, model, eval_env, args.train)
        if args.save:
            print('Saving model', file=sys.stderr)
            model.save(args.save)
            print('Saving normalization statistics', file=sys.stderr)
            model.get_vec_normalize_env().save('%s.norm' % args.save)

    elif args.run:
        print('Running the model', file=sys.stderr)
        ep_rewards, ep_lengths = sb3.common.evaluation.evaluate_policy(
            model, run_env, n_eval_episodes=1, return_episode_rewards=True)
        run_env.env_method('dump')

        print('-' * 80, file=sys.stderr)
        print('Total episode length: %d timesteps' % ep_lengths[0], file=sys.stderr)
        print('Total agent reward during episode: %.2f' % ep_rewards[0], file=sys.stderr)

        (initial_score, final_score) = run_env.env_method('get_score_diff')[0]
        impr = final_score - initial_score
        impr_pct = 100 * impr / initial_score

        print('Trial score at start of episode: %.2f' % initial_score, file=sys.stderr)
        print('Trial score at end of episode: %.2f' % final_score, file=sys.stderr)
        print('Score improvement by agent: %.2f (%.2f%%)' % (impr, impr_pct), file=sys.stderr)
