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
# simulator.py - Interface to the network simulator
#
# This module implements the interface between the network simulator (written
# in C++ and hosted in a separate process) and the RL agent. We also use the
# same interface when running optimization using grid search and Optuna. This
# file is thus non-RL-specific, while the RL parts are kept in environment.py

import math
import random
import re
import subprocess
import sys

UE, CELL, RUN = 'ue', 'cell', 'run'

class SimulatorInterface:
    '''
    Base class for our interface to the network simulator, responsible for
    parsing of time series data from the simulator (related to particular UEs
    or cells), sending actions back to the simulator, and calculating the score
    value for a given trial. Subprocess handling and interaction is done in a
    separate subclass SimulatorSubprocess. This way, we can also reuse this
    base class when rendering/plotting previous trials, by "replaying" the
    output log into the same parsing/scoring logic (see SimulatorLogReplay)
    '''

    def __init__(self):
        self.timestep = None
        self.current_seed = None
        self.initial_score = None

        # All time series data parsed from the simulator is kept in self.data,
        # a nested dict keyed on data[obj_type][obj_id][timeseries_name]. Each
        # entry contains a sorted list of (timestep, value) tuples, with
        # obj_type as either 'ue', 'cell' or 'run' (the latter referring to
        # time series that apply to the whole simulation, e.g. the game score)
        self.data = { UE: {}, CELL: {}, RUN: {} }

    def get_next_output_line(self):
        '''
        Get the next output line from the simulator. This method should be
        overridden by the subclass, to either implement an interface to a live,
        running subprocess (SimulatorSubprocess), or to implement parsing of
        the log from a past trial (SimulatorLogReplay)
        '''
        pass

    def run_parse_loop(self, break_on_interaction=False, periodic_callback=None, callback_interval=100):
        while True:
            # Get next line. Bail if empty (likely means subprocess terminated)
            line = self.get_next_output_line()
            if len(line) == 0:
                return False

            # Every line from the simulator is of the form "[timestep] ms: [...]".
            # Start by extracting timestep before parsing the rest of the event
            match = re.match(r'^(-?\d+) ms: (.+)$', line)
            if match is None:
                continue
            line_timestep = int(match.group(1))
            line_content = match.group(2)

            # If we are about to enter a new cycle of the callback interval,
            # perform the callback before proceeding with parsing the new line.
            # This callback is used for plotting at 0 ms, 100 ms, 200 ms, etc.
            if self.timestep is None:
                self.timestep = line_timestep
            last_period = ((self.timestep - 1) // callback_interval)
            next_period = ((line_timestep - 1) // callback_interval)
            if periodic_callback is not None and last_period != next_period:
                # Make the timestep appear as the final timestep of the last period
                self.timestep = (last_period + 1) * callback_interval
                periodic_callback()
            self.timestep = line_timestep

            # Parse out the interesting events from the simulation output line
            self.parse_event(line_content)

            # We may want to quit looping when we reach the interaction point
            if re.match(r'^Agent action\?$', line_content):
                # If at the first interaction point (meaning the simulator just
                # finished the warmup phase), then save the initial trial score
                if self.initial_score is None:
                    self.initial_score = self.calculate_score()
                # If asked to break the loop at the interaction point, do so now
                if break_on_interaction:
                    return True

    def add_data(self, obj_type, obj_id, **d):
        if obj_id not in self.data[obj_type]:
            self.data[obj_type][obj_id] = {}
        for key, value in d.items():
            if key not in self.data[obj_type][obj_id]:
                self.data[obj_type][obj_id][key] = []
            # Append (timestep, value) tuples to a list such as data['ue'][2]['bytes_rx']
            self.data[obj_type][obj_id][key].append((self.timestep, value))

    def get_objects(self, obj_type):
        return sorted(self.data[obj_type].keys())

    def timeseries_valid(self, obj_type, obj_id, key):
        return (obj_type in self.data) and (obj_id in self.data[obj_type]) and (key in self.data[obj_type][obj_id])

    def get_timeseries(self, obj_type, obj_id, key):
        return self.data[obj_type][obj_id][key]

    def get_value(self, obj_type, obj_id, key, else_val=None):
        if not self.timeseries_valid(obj_type, obj_id, key):
            return else_val
        # Take the latest (timestep, value) tuple and return only the value
        timeseries = self.get_timeseries(obj_type, obj_id, key)
        return timeseries[-1][1] if len(timeseries) > 0 else else_val

    def get_last_n_values(self, obj_type, obj_id, key, n):
        timeseries = self.get_timeseries(obj_type, obj_id, key)
        return [ t[1] for t in timeseries[-n:] ]

    def parse_event(self, event):
        match = re.match(r'^Seed (.+)$', event)
        if match:
            self.current_seed = match.group(1)
        match = re.match(r'^Cell state: Cell (.+) at (.+) (.+) direction (.+)$', event)
        if match:
            self.add_data(CELL, int(match.group(1)),
                coords=(float(match.group(2)), float(match.group(3))),
                direction=float(match.group(4)))
        match = re.match(r'^UE state: IMSI (.+) at (.+) (.+) with (.+) received bytes$', event)
        if match:
            self.add_data(UE, int(match.group(1)),
                coords=(float(match.group(2)), float(match.group(3))),
                bytes_rx=int(match.group(4)))
        match = re.match(r'^UE seen at cell: Cell (.+) saw IMSI (.+)$', event)
        if match:
            self.add_data(UE, int(match.group(2)),
                cell_associated=int(match.group(1)))
        match = re.match(r'^Measurement report: Cell .+ got report from IMSI (.+): (.+)$', event)
        if match:
            imsi = int(match.group(1))
            measurements = match.group(2).split(' ')
            for measurement in measurements:
                match = re.match(r'^(.+)/(.+)/(.+)$', measurement)
                cell, rsrp, rsrq = int(match.group(1)), int(match.group(2)), int(match.group(3))
                self.add_data(UE, imsi, **{
                    'rsrp_for_%d' % cell: rsrp,
                    'rsrq_for_%d' % cell: rsrq,
                })
        match = re.match(r'^Configuration: Cell tx1 (.+) tx2 (.+) tx3 (.+)$', event)
        if match:
            self.add_data(CELL, 1, tx_power=int(match.group(1)))
            self.add_data(CELL, 2, tx_power=int(match.group(2)))
            self.add_data(CELL, 3, tx_power=int(match.group(3)))
        match = re.match(r'^Configuration legality: Cell tx1 (.+) tx2 (.+) tx3 (.+)$', event)
        if match:
            self.add_data(CELL, 1, tx_legal=bool(int(match.group(1))))
            self.add_data(CELL, 2, tx_legal=bool(int(match.group(2))))
            self.add_data(CELL, 3, tx_legal=bool(int(match.group(3))))

    def calculate_score_for_ue(self, ue):
        '''
        Calculate experience score for a given user's experienced throughput in
        the past 2 seconds. There is a diminishing return, logarithmically
        approaching a score of 1 at 2 Mbps (= 500000 bytes over a 2 sec window)
        '''
        thruput = sum(self.get_last_n_values(UE, ue, 'bytes_rx', 20))
        return math.log(thruput * 999 / 500000 + 1, 1000)

    def calculate_score(self):
        '''
        Calculate the total trial score across all users, based on their
        throughputs in the past 2 seconds and using the diminishing returns
        implemented in calculate_score_for_ue()
        '''
        return sum(self.calculate_score_for_ue(ue)
            for ue in self.get_objects(UE))

class SimulatorSubprocess(SimulatorInterface):
    def __init__(self, levels, ue_count=12, initial_parameters=None, mock=False, randomize=False, quiet=False):
        # Determine current level and the initial base station parameters
        current_level = levels[random.randrange(len(levels))]
        if initial_parameters is None:
            initial_parameters = [300, 300, 300]
        if randomize:
            initial_parameters = [ random.randint(250, 350) for i in range(3) ]

        # Assemble the command line arguments for the simulator subprocess
        process_args = [
            './simulator',
            '--seeds=%s' % current_level,
            '--left=%d' % initial_parameters[0],
            '--right=%d' % initial_parameters[1],
            '--top=%d' % initial_parameters[2],
            '--ue_count=%d' % ue_count,
        ]
        if mock:
            process_args.append('--mock')

        # Spawn the simulator subprocess, with accompanying stdin/stdout pipes
        self.process = subprocess.Popen(process_args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            universal_newlines=True, bufsize=1)
        self.output_log = []

        # Print debug output indicating that the simulator is warming up
        if not quiet:
            print('[%5d] Simulator is warming up (level %s)...' % \
                (self.process.pid, current_level), file=sys.stderr)
        super().__init__()

    def get_next_output_line(self):
        line = self.process.stdout.readline()
        # Log the output from the subprocess, in case we later want to save it
        self.output_log.append(line)
        return line

    def send_action(self, action):
        # Assemble the action string as expected by the simulator program
        action_string = '%d %d %d' % tuple(action)
        # Send it over the standard input pipe to the simulator subprocess
        self.process.stdin.write(action_string + '\n')
        self.process.stdin.flush()

    def quit(self):
        self.process.kill()

class SimulatorLogReplay(SimulatorInterface):
    def __init__(self, output_log):
        self.output_log = output_log
        self.current_line = 0
        super().__init__()

    def get_next_output_line(self):
        if self.current_line >= len(self.output_log):
            return ''
        self.current_line += 1
        return self.output_log[self.current_line - 1]
