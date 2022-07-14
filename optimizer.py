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
# optimizer.py - Helper code for Optuna optimization
#
# This module integrates our network simulator pipeline with the Optuna library
# for black-box optimization. We use Optuna to find good "baseline" solutions
# for given levels in the network scenario, in order to have a comparison for
# the RL agent. Optuna is also used for grid search, but then mainly as a data
# store for the trial results. The outcome of trials conducted using either
# grid search, Optuna search or RL are all stored in Optuna's SQLite based data
# store, so our plotting and tabulation code can be reused across all modes

import argparse
import itertools
import json
import math
import multiprocessing
import numpy as np
import optuna
from optuna.trial import TrialState
import pandas as pd
import random
import sys
import time
import warnings
warnings.filterwarnings('ignore', r'.*_trial is experimental.*')

import simulator
import visualizer

class Optimizer:
    def __init__(self, db='sqlite:///scorecard.db'):
        self.db = db
        self.level_stats = {}
        self.optuna_trial_count = 125

        # References to salient trials we might want to plot for the paper
        self.baseline_trials = {}
        self.rl_median_trials = {}

    # Explicitly state the values to visit during grid search
    grid_search_space = {
        'tx1': [ 200, 250, 300, 350, 400 ],
        'tx2': [ 200, 250, 300, 350, 400 ],
        'tx3': [ 200, 250, 300, 350, 400 ],
    }

    def perform_trial(self, trial):
        '''
        Helper method to perform a single trial for Optuna optimization. Spawns
        a simulator with inital network parameters sampled by Optuna. After the
        simulator warmup phase has finished, we capture the simulation output
        and return the final trial score as calculated by the simulator module
        '''

        # Sample the initial network parameters from Optuna
        parameters = [
            trial.suggest_int('tx1', 200, 400),
            trial.suggest_int('tx2', 200, 400),
            trial.suggest_int('tx3', 200, 400),
        ]
        # Set up the simulator with the sampled initial parameters
        trial_simulator = simulator.SimulatorSubprocess(
            initial_parameters=parameters, **self.simulator_kwargs)
        # Run simulator until first interaction, which means warmup is done
        trial_simulator.run_parse_loop(break_on_interaction=True)
        # Return zero score if simulator never finished the warmup phase
        if trial_simulator.timestep < 0:
            return 0
        # Kill the simulator subprocess
        trial_simulator.quit()
        # Capture simulation output and store in Optuna's database
        trial.set_user_attr('output_log', trial_simulator.output_log)
        # Return trial score as determined by the simulator module
        return trial_simulator.calculate_score()

    def delete_study_if_exists(self, study_type, level):
        try:
            optuna.delete_study('%s-%d' % (study_type, level), self.db)
        except KeyError:
            pass

    def get_or_create_study(self, study_type, level, mock=False):
        study_sampler = {
            'grid': optuna.samplers.GridSampler(self.grid_search_space),
            'optuna': optuna.samplers.TPESampler(),
            'rl': None,
        }[study_type]
        study = optuna.create_study(storage=self.db,
            study_name='%s-%d' % (study_type, level), load_if_exists=True,
            sampler=study_sampler, direction='maximize')
        if len(study.user_attrs) == 0:
            study.set_user_attr('study_type', study_type)
            study.set_user_attr('level', level)
            study.set_user_attr('mock', mock)
        return study

    def run_study(self, study, parallelism=1):
        study_n_trials = len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]))
        target_n_trials = {
            'grid': 999999,
            'optuna': self.optuna_trial_count,
        }[study.user_attrs['study_type']]
        remaining_n_trials = target_n_trials - study_n_trials
        if remaining_n_trials > 0:
            self.simulator_kwargs = {
                'levels': [study.user_attrs['level']],
                'mock': study.user_attrs['mock'],
            }
            our_n_trials = math.ceil(remaining_n_trials / parallelism)
            study.optimize(self.perform_trial, n_trials=our_n_trials)

    # The remainder of this class contains methods that can be used to plot,
    # visualize and tabulate results that are already stored in an Optuna
    # database. These methods are meant to be used from a Jupyter notebook

    def render_trial(self, trial):
        '''
        Method to render the state of the envionment at the end of an
        optimization trial. Fetches the stored output log from the Optuna
        database and passes it to the visualization module
        '''
        import IPython.display
        output_log = trial.user_attrs['output_log']
        png_data = visualizer.SimulationRenderer.render_final_frame(output_log)
        IPython.display.display_png(png_data, raw=True)

    def export_plots(self, trial, label='', path_prefix=''):
        '''
        Method to export clean parts of the "main" plot (such as the top-down
        view) to standalone PDF files, in order to make figures for papers etc.
        '''
        output_log = trial.user_attrs['output_log']
        for plot_type in [ 'map', 'config' ]:
            pdf_data = visualizer.SimulationRenderer.render_final_frame(
                output_log, template=plot_type, label=label)
            filename = '%s%s.pdf' % (path_prefix, plot_type)
            with open(filename, 'wb') as f:
                f.write(pdf_data)

    def show_grid_search_results(self, study, plotting=True):
        space = self.grid_search_space
        dims = sorted(space.keys())
        grid = [[[ 0.0 for dim2_val in space[dims[2]] ]
                for dim1_val in space[dims[1]] ]
            for dim0_val in space[dims[0]] ]

        all_results = []
        baseline_result = 0.0
        for trial in study.get_trials(deepcopy=False, states=[TrialState.COMPLETE]):
            t = tuple(space[dim].index(trial.params[dim]) for dim in dims)
            grid[t[0]][t[1]][t[2]] = cur_result = trial.values[0]
            all_results.append(cur_result)
            if trial.params['tx1'] == trial.params['tx2'] == trial.params['tx3'] == 300:
                # Consider the trial where all base stations are transmitting
                # at 30.0 dBm as the "baseline" trial. We assume the network
                # operator would leave the base stations configured to these
                # settings, and this is thus considered the baseline result
                # with which to compare grid search and Optuna
                baseline_result = cur_result
                self.baseline_trials[study.user_attrs['level']] = trial

        if plotting:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6 * len(space[dims[0]]), 4))
            plt.style.use('default')
            for i, dim0_val in enumerate(space[dims[0]]):
                plt.subplot(1, len(space[dims[0]]), i + 1)
                plt.xticks(range(len(space[dims[1]])), space[dims[1]])
                plt.xlabel(dims[1])
                plt.yticks(range(len(space[dims[2]])), space[dims[2]])
                plt.ylabel(dims[2])
                plt.ylim(-0.5, len(space[dims[2]]) - 0.5)
                plt.imshow(np.array(grid[i]).T)
                plt.clim(min(all_results), max(all_results))
                plt.title('%s = %s' % (dims[0], str(dim0_val)))
                plt.colorbar()
            plt.show()
        return baseline_result

    def show_level_results(self, level, plotting=True):
        import IPython.display
        import matplotlib.pyplot as plt
        import warnings
        warnings.filterwarnings('ignore')

        IPython.display.display_markdown('# Level %d' % level, raw=True)
        stats = {}
        for study_type in [ 'grid', 'optuna', 'rl' ]:
            IPython.display.display_markdown('## Optimization results from %s' % study_type, raw=True)
            study = self.get_or_create_study(study_type, level)
            if len(study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])) == 0:
                IPython.display.display_markdown('No trials found in database', raw=True)
                continue
            if study_type == 'grid':
                stats['baseline'] = self.show_grid_search_results(study, plotting)
            if study_type != 'rl' and plotting:
                optuna.visualization.matplotlib.plot_optimization_history(study)
                plt.show()
            if plotting:
                self.render_trial(study.best_trial)
            stats[study_type] = study.best_value
            if 'baseline' in stats:
                stats['%s / baseline' % study_type] = stats[study_type] / stats['baseline'] - 1.0
            if study_type == 'optuna' and 'grid' in stats:
                stats['optuna / grid'] = stats['optuna'] / stats['grid'] - 1.0
            if study_type == 'rl' and 'grid' in stats:
                stats['rl / grid'] = stats['rl'] / stats['grid'] - 1.0
            if study_type == 'rl' and 'optuna' in stats:
                stats['rl / optuna'] = stats['rl'] / stats['optuna'] - 1.0

        IPython.display.display_markdown('## Summary', raw=True)
        self.show_results_table(pd.DataFrame(stats, index=['Level %d' % level]))
        self.level_stats['Level %d' % level] = stats

    def show_results_table(self, table):
        formatted_table = pd.DataFrame(table, copy=True)
        for col in formatted_table.columns:
            if col in [ 'baseline', 'grid', 'optuna', 'rl' ]:
                formatted_table[col] = formatted_table[col].apply(lambda x: '' if (x != x) else '%.2f' % x)
            if col in [ 'grid / baseline', 'optuna / baseline', 'optuna / grid', 'rl / baseline', 'rl / grid', 'rl / optuna' ]:
                formatted_table[col] = formatted_table[col].apply(lambda x: '%.1f%%' % (x * 100.0))
        import IPython.display
        IPython.display.display(formatted_table)

    def show_final_results_table(self):
        import IPython.display
        IPython.display.display_markdown('# Final results across all levels', raw=True)
        table = pd.DataFrame(self.level_stats).T
        cols = [ 'grid / baseline', 'optuna / baseline', 'optuna / grid', 'rl / baseline', 'rl / grid', 'rl / optuna' ]
        table = table.append(pd.DataFrame({ c: table[c].mean() for c in cols if c in table.columns }, index=['Mean']))
        table = table.append(pd.DataFrame({ c: table[c].median() for c in cols if c in table.columns }, index=['Median']))
        self.show_results_table(table)

    def get_all_levels(self):
        study_summaries = optuna.study.get_all_study_summaries(self.db)
        return list(sorted(set(study.user_attrs['level'] for study in study_summaries)))

    def show_all_results(self, plotting=True):
        for level in self.get_all_levels():
            self.show_level_results(level, plotting)
        self.show_final_results_table()

    def generate_rl_scorecard(self, levels, pdf_filename=None):
        # Fetch all the RL trial results into a handy table
        rl_results = {}
        for (level, label) in levels:
            study = self.get_or_create_study('rl', level)
            trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            rl_results[level] = [ trial.values[0] for trial in trials ]

            # Find the median RL trial and store it
            sorted_trials = [ (trial.values[0], trial) for trial in trials ]
            sorted_trials.sort()
            median_trial = sorted_trials[len(sorted_trials) // 2][1]
            self.rl_median_trials[level] = median_trial

        # Import matplotlib and reset the style settings
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.rcdefaults()

        # Render the scorecard, level by level
        plt.figure(figsize=(5, len(levels)))
        for row, (level, label) in enumerate(levels):
            # Plot the "benchmark markers"
            for marker_key, marker_color, tick_type, tick_delta, label in [
                    ('optuna','darkgreen','v',-1, 'Best Optuna TPE trial'),
                    ('grid','orange','o',1, 'Best grid search trial'),
                    ('baseline', 'maroon','^',1, 'Baseline (default settings)')]:
                marker_value = self.level_stats['Level %d' % level][marker_key]
                plt.plot([marker_value,marker_value], [row - 0.275, row + 0.275], c=marker_color)
                plt.scatter([marker_value], [row + 0.35 * tick_delta],
                    c=marker_color, s=100, marker=tick_type, edgecolors='none',
                    label=label if row==0 else None)

            # Plot a "stripplot" of all the indidual RL trial results
            plt.scatter(
                rl_results[level],
                [row + 0.1 * (random.random() * 2 - 1) for _ in rl_results[level] ],
                c='steelblue', alpha=0.25, s=100, edgecolors='none',
                label='Individual RL trial' if row==0 else None)

            # Plot a bold star for the median RL score
            plt.scatter(
                [pd.Series(rl_results[level]).median()], [row],
                c='black', s=250, marker='*', edgecolors='white', zorder=10,
                label='Median RL score' if row==0 else None)

        # Finalize plot, save to PDF and display it
        plt.ylim(len(levels) - 0.5, -0.5)
        plt.grid(True, axis='y')
        plt.gca().set_axisbelow(True)
        plt.box()
        plt.tick_params('y', length=0)
        plt.yticks(range(len(levels)), [label for (level, label) in levels], fontweight='bold')
        plt.xlabel('User experience score', fontweight='bold')
        plt.figlegend(loc='lower center', ncol=2)
        plt.title('RL agent scorecard', fontweight='bold')
        plt.tight_layout(pad=0, rect=(0, 0.15, 1, 1))
        if pdf_filename is not None:
            plt.savefig(pdf_filename)
        plt.show()

def spawn_study(args, study_type, level):
    optimizer = Optimizer(args.db)
    study = optimizer.get_or_create_study(study_type, level, args.mock)
    optimizer.run_study(study, args.parallelism)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='sqlite:///scorecard.db', help='SQLite database file to store trial data')
    subparsers = parser.add_subparsers(dest='action')
    subparser = subparsers.add_parser('import')
    subparser.add_argument('logs', nargs='+', help='Files containing simulation logs to import to the database')
    subparser = subparsers.add_parser('export')
    subparser.add_argument('study_type', choices=['grid','optuna','rl'], help='Which study type to fetch trial from')
    subparser.add_argument('level', type=int, help='Which simulator level to fetch trial from')
    subparser = subparsers.add_parser('results')
    subparser.add_argument('output', help='File into which to dump final results (as JSON)')
    subparser = subparsers.add_parser('optimize')
    subparser.add_argument('--mock', action='store_true', help='Use the mock version of the simulator')
    subparser.add_argument('--parallelism', type=int, default=1, help='How many parallel jobs to spawn per study')
    subparser.add_argument('--grid', action='store_true', help='Perform grid search and store results in the database')
    subparser.add_argument('--optuna', action='store_true', help='Perform Optuna search and store results in the database')
    subparser.add_argument('levels', nargs='+', type=int, help='Which simulator levels to run optimization on')
    args = parser.parse_args()

    if args.action == 'import':
        # Keep track of RL studies already emptied out in this import session
        already_emptied_studies = []
        # Loop through all the log files given on the command line
        optimizer = Optimizer(args.db)
        for log in args.logs:
            # Parse the output log, to be able to calculate the trial score
            with open(log, 'r') as f:
                lines = f.readlines()
            replay_simulator = simulator.SimulatorLogReplay(lines)
            replay_simulator.run_parse_loop()

            # Store the trial in the Optuna database, under a separate "RL
            # study". If such a study already exists and it hasn't yet been
            # "emptied out" in this import session, then do that first
            level = int(replay_simulator.current_seed)
            if level not in already_emptied_studies:
                optimizer.delete_study_if_exists('rl', level)
                already_emptied_studies.append(level)
            study = optimizer.get_or_create_study('rl', level)
            study.add_trial(optuna.trial.create_trial(
                value=replay_simulator.calculate_score(),
                user_attrs={ 'output_log': lines }))

    if args.action == 'export':
        # Export the output log from a trial stored in the Optuna database
        optimizer = Optimizer(args.db)
        study = optimizer.get_or_create_study(args.study_type, args.level)
        print(''.join(study.best_trial.user_attrs['output_log']))

    if args.action == 'results':
        # Calculate the final results table and save it to JSON
        optimizer = Optimizer(args.db)
        optimizer.show_all_results(plotting=False)
        with open(args.output, 'w') as f:
            json.dump(optimizer.level_stats, f, indent=2)
        print('\nResults saved to %s' % args.output)

    if args.action == 'optimize':
        # A "study" is defined by a search type (grid search or Optuna search) and
        # a level we want to optimize using that search type. Each study runs in a
        # separate parallel processes, using the multiprocessing module below. The
        # user can also specify additional parallelism using a multiplier
        study_types = []
        if args.grid: study_types += [ 'grid' ] * args.parallelism
        if args.optuna: study_types += [ 'optuna' ] * args.parallelism
        if len(study_types) == 0:
            print('Specify at least one of --grid and --optuna', file=sys.stderr)
            sys.exit()

        print('Initializing the Optuna database by creating the first study before spawning workers')
        studies = list(itertools.product(study_types, args.levels))
        Optimizer(args.db).get_or_create_study(studies[0][0], studies[0][1], args.mock)

        print('Spawning %d workers assigned to the following studies:' % len(studies), end=' ')
        print(' '.join(map(lambda t: '%s-%d' % t, studies)))
        pool = multiprocessing.Pool(len(studies))
        pool.starmap(spawn_study, [ (args, study_type, level) for (study_type, level) in studies ])
