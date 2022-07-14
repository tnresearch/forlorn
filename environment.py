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
# environment.py - Implementation of Gym environment
#
# This module implements our Gym environment for the network optimization
# problem, building on top of the simulator interface in simulator.py. This
# file contains everything related to observation spaces, actions spaces,
# reward design and game-over conditions, i.e. the parts of our experiment that
# are specific to Reinforcement Learning. The non-RL-specific parts, such as
# the calculation of a "trial score" for an episode, are kept in simulator.py

import gym
import gym.spaces
import numpy as np
import sys

import simulator
UE, CELL = 'ue', 'cell'

class Environment(gym.Env):
    def __init__(self, duration, history=1, step_size=10, quiet=False, num_rsrq_quantiles=None, rsrq_quantile_fractions=None, oob_penalty_factor=None, oob_means_gameover=False, **simulator_kwargs):
        # Determine the RSRQ quantiles to include in the observation space
        # (either given explicity or as a number of uniformly spaced quantiles)
        if rsrq_quantile_fractions is None:
            if num_rsrq_quantiles is None:
                num_rsrq_quantiles = 1
            rsrq_quantile_fractions = [float(q)/float(num_rsrq_quantiles) for q in range(1, num_rsrq_quantiles)]
        else:
            if num_rsrq_quantiles is not None:
                print('--num-rsrq-quantiles and --rsrq-quantile-fractions are incompatible', file=sys.stderr)
                sys.exit(1)
        self.rsrq_quantile_fractions = rsrq_quantile_fractions

        self.oob_penalty_factor = oob_penalty_factor
        self.oob_means_gameover = oob_means_gameover

        # Agent observation is a (flattened) 3D tensor with a stack of recent
        # observations for the last N timesteps (given by the history argument
        # to the constructor).  The inner matrix (last two dimensions) has 3
        # rows (1 per cell), each with 2 (optionally more) values:
        #   - Current transmission power
        #   - Number of connected UEs
        #   - RSRQ quantiles
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(history * 3 * (2 + len(rsrq_quantile_fractions)),))

        # Agent action space consists of 7 discrete choices:
        #     0: No-op
        #   1-2: Decrease/increase transmission power for cell 1
        #   3-4: Decrease/increase transmission power for cell 2
        #   5-6: Decrease/increase transmission power for cell 3
        self.action_space = gym.spaces.Discrete(7)

        # Various other configuration for the simulator and the environment
        self.game_duration_ms = duration
        self.history_steps = history
        self.step_size = step_size
        self.quiet = quiet
        self.simulator_kwargs = simulator_kwargs
        self.simulator_kwargs['quiet'] = self.quiet

        # Clear the variables that are assumed to be set in reset()
        self.debug_string = ''
        self.simulator = None
        self.next_simulator = None

    def reset(self):
        # If we are already in a freshly reset state, don't bother again
        if self.simulator is not None and self.simulator.timestep == 0:
            return self.observation

        # If there was a previous episode in this environment, dump out the
        # debugging string together with the associated game-over condition
        if len(self.debug_string) > 0 and not self.quiet:
            reason = 'external reset' if self.done_reason is None else self.done_reason
            print('%s: Game over (%s)' % (self.debug_string, reason), file=sys.stderr)

        # Clear the variables that are maintained by process_simulator_output()
        self.observation = None
        self.reward = None
        self.done = False
        self.done_reason = None
        self.last_score = None

        # If there was a previous episode, terminate the old simulator
        if self.simulator is not None:
            self.simulator.quit()

        # Set up the simulator. We use "double buffering" of simulators, to
        # avoid delaying the training process when restarting the environment.
        # If this is the first startup of the environment, we don't have a
        # standby simulator ready, thus we explicitly need to create it now
        if self.next_simulator is None:
            self.next_simulator = simulator.SimulatorSubprocess(**self.simulator_kwargs)

            # The output log from the environment's first simulator is special,
            # as this is the log we want to return from the dump() method
            self.output_log = self.next_simulator.output_log

        # Swap in the standby simulator as the current one, and spawn the next one
        self.simulator = self.next_simulator
        self.next_simulator = simulator.SimulatorSubprocess(**self.simulator_kwargs)

        # Parse initial process output and return the first observation to the agent
        self.process_simulator_output()
        return self.observation

    def step(self, action):
        # Apply the agent's "differential action" to the current cell power
        # configuration contained in self.tx_powers. This internal variable is
        # populated/updated by process_simulator_output() every timestep
        if action == 1: self.tx_powers[0] -= self.step_size
        if action == 2: self.tx_powers[0] += self.step_size
        if action == 3: self.tx_powers[1] -= self.step_size
        if action == 4: self.tx_powers[1] += self.step_size
        if action == 5: self.tx_powers[2] -= self.step_size
        if action == 6: self.tx_powers[2] += self.step_size

        # Send the updated self.tx_powers to the simulator as the next action
        self.simulator.send_action(self.tx_powers)

        # Print out the current status debug line and the action taken
        if not self.quiet:
            action_name = ['--','1-','1+','2-','2+','3-','3+'][action]
            print('%s: Action %s, config %s' % \
                (self.debug_string, action_name, str(self.tx_powers)), file=sys.stderr)

        # Keep a reference to the last simulator that the environment actually
        # interacted with (needed because Gym will reset the environment at the
        # end of a trial, thus erasing our main reference to self.simulator)
        self.last_simulator = self.simulator

        # Parse simulator output and return the (observation, reward, done, info) tuple
        self.process_simulator_output()
        return (self.observation, self.reward, self.done, {})

    def process_simulator_output(self):
        # Parse simulator output until the interaction point. If this returns
        # false, the simulator has terminated, so return dummy values
        if not self.simulator.run_parse_loop(break_on_interaction=True):
            self.observation = np.empty(self.observation_space.shape)
            self.reward = 0.0
            self.done = True
            self.done_reason = 'simulator crashed'
            return

        # Determine the latest transmission power settings on each of the cells
        self.tx_powers = [
            self.simulator.get_value(CELL, cell, 'tx_power')
            for cell in self.simulator.get_objects(CELL) ]

        # Determine whether the agent's last choice of transmission powers was
        # legal.
        self.tx_legal = [
            self.simulator.get_value(CELL, cell, 'tx_legal')
            for cell in self.simulator.get_objects(CELL) ]

        # Count the number of UEs attached to each cell
        self.ue_counts = [ 0 ] * len(self.simulator.get_objects(CELL))
        for ue in self.simulator.get_objects(UE):
            cell = self.simulator.get_value(UE, ue, 'cell_associated')
            if cell is not None:
                self.ue_counts[cell - 1] += 1

        # Calculate UE throughput from a 1 second (10 timesteps) sliding window
        self.ue_thruputs = [
            sum(self.simulator.get_last_n_values(UE, ue, 'bytes_rx', 10)) * 8 / 1000
            for ue in self.simulator.get_objects(UE) ]

        # Collect signal quality (RSRQ) statistics for each cell, and
        # extract the desired quantiles as features
        rsrqs = []
        for cell in self.simulator.get_objects(CELL):
            rsrqs.append([])
            for ue in self.simulator.get_objects(UE):
                connected_cell = self.simulator.get_value(UE, ue, 'cell_associated')
                if connected_cell == cell and self.simulator.timeseries_valid(UE, ue, 'rsrq_for_%d' % cell):
                    rsrqs[-1].append(self.simulator.get_value(UE, ue, 'rsrq_for_%d' % cell))
            if len(rsrqs[-1]) == 0:
                rsrqs[-1].append(0.0)
        rsrq_quantiles = [np.quantile(rsrqs[cell], self.rsrq_quantile_fractions) for cell in range(3)]

        # Assemble the observation input matrix for the agent's neural network
        cur_obs = sum([[
                self.tx_powers[cell],
                self.ue_counts[cell],
                *(rsrq_quantiles[cell]),
            ] for cell in range(3) ], start=[])
        if self.observation is None:
            self.observation = np.array(cur_obs * self.history_steps, dtype=np.float32)
        self.observation = np.roll(self.observation, len(cur_obs))
        self.observation[:len(cur_obs)] = cur_obs

        # Calculate reward, based on the change in trial score since last timestep
        current_score = self.simulator.calculate_score()
        if self.last_score is None:
            # Initialize last_score to initial current_score value, to get zero
            # instead of an arbitrary, exaggerated reward in the first timestep
            self.last_score = current_score
        self.reward = current_score - self.last_score
        self.last_score = current_score

        # The game is over when both the warmup and game phases have passed
        if self.simulator.timestep >= self.game_duration_ms:
            self.done = True
            self.done_reason = 'episode finished'

        # We attempted setting transmission powers out of bounds in
        # our last action.
        if False in self.tx_legal:
            # Game over.
            if self.oob_means_gameover:
                self.done = True
                self.done_reason = 'out of bounds'
            # Penalize. The size of the penalty is equal to the number
            # of users, so its magnitude is as large as the reward for
            # "good enough" service (2 Mbps, reward 1.0 per user).
            self.reward += -self.oob_penalty_factor*sum(self.ue_counts)

        # Assemble debug string to be printed out alongside the action taken in step()
        self.debug_string = '[%5d]' % self.simulator.process.pid
        self.debug_string += ' ' + ', '.join([
            'Level %s' % self.simulator.current_seed.rjust(4),
            'time %5d' % self.simulator.timestep,
            'min %5d kbps' % min(self.ue_thruputs),
            'max %5d kbps' % max(self.ue_thruputs),
            'score %5.2f' % self.last_score,
            'UEs %s' % ' '.join(map(lambda c: '%2d' % c, self.ue_counts)),
            'legal %s' % ''.join(map(lambda l: 'T' if l else 'F', self.tx_legal)),
            'reward % .2f' % self.reward,
        ])

    def dump(self):
        # Dump the captured output from the simulator. Used by e.g. trainer.py
        # so that we can render animations of the behavior of an agent. We are
        # only interested in the output from the first instantiation of the
        # simulator, so we use the variable self.output_log (which is assigned
        # on the first reset in method reset() above)
        for line in self.output_log:
            sys.stdout.write(line)
        sys.stdout.flush()
        self.output_log.clear()

    def get_score_diff(self):
        # Used by trainer.py to calculate final stats when running the agent
        initial_score = self.last_simulator.initial_score
        final_score = self.last_simulator.calculate_score()
        return (initial_score, final_score)
