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
# visualizer.py - Visualization of the simulation state
#
# This module takes an output log from a previous run of the simulator and
# visualizes the simulation state. The module reuses the output parser from the
# simulator interface in simulator.py, and uses Gnuplot for plot rendering

import argparse
import json
import math
import os
import subprocess
import sys

import simulator
UE, CELL, RUN = 'ue', 'cell', 'run'

class GnuplotInterface:
    '''
    Helper class to quickly build plots using Gnuplot. The class communicates
    with the external Gnuplot process using the subprocess module. There are
    helper methods to keep track of the "plot elements" to include in the plot,
    which are 1D/2D data series with an accompanying plot definition that tells
    Gnuplot e.g. whether to do line, area or scatter plot, color settings etc.
    Start `gnuplot` in interactive mode from the terminal and run `help plot`
    for more information. There are also helper methods to add "objects", such
    as geometrical shapes, labels and arrows to the plot.
    '''

    def __init__(self):
        self.gnuplot_instance = None
        self.plot_objects = []
        self.reset_plot_elements()

    def reset_plot_elements(self):
        self.plot_elements = [ ('1/0 notitle', '') ]

    def add_plot_element(self, definition, inline_data=None):
        # For performance reasons, build the string piece-by-piece
        serialized_inline_data = ''
        if inline_data is not None:
            first_row = True
            for row in inline_data:
                if not first_row:
                    serialized_inline_data += '\n'
                first_val = True
                for val in row:
                    if not first_val:
                        serialized_inline_data += ' '
                    serialized_inline_data += str(val)
                    first_val = False
                first_row = False
            serialized_inline_data += '\ne\n'
        self.plot_elements.append((definition, serialized_inline_data))

    def finish_plot_command(self):
        plot_command = self.get_plot_objects() + '\n'
        plot_command += 'plot ' + ', '.join(pe[0] for pe in self.plot_elements) + ';\n'
        plot_command += ''.join(pe[1] for pe in self.plot_elements)
        self.reset_plot_elements()
        return plot_command

    def add_object(self, definition, *args):
        self.plot_objects.append('set object %d %s;' % (len(self.plot_objects) + 1, definition % args))

    def add_label(self, definition, *args):
        self.plot_objects.append('set label %s;' % (definition % args))

    def add_arrow(self, definition, *args):
        self.plot_objects.append('set arrow %s;' % (definition % args))

    def get_plot_objects(self):
        serialized_plot_objects = '\n'.join(self.plot_objects)
        self.plot_objects = []
        return serialized_plot_objects

    def run_commands(self, plot_commands):
        if self.gnuplot_instance is None:
            self.gnuplot_instance = subprocess.Popen('gnuplot',
                stdin=subprocess.PIPE)
        self.gnuplot_instance.stdin.write(plot_commands.encode('utf-8'))
        self.gnuplot_instance.stdin.flush()

    @staticmethod
    def render_single_frame(plot_commands):
        result = subprocess.run('gnuplot',
            input=plot_commands.encode('utf-8'),
            capture_output=True)
        return result.stdout

FRAME_TEMPLATE = '''
set terminal pngcairo size 1600,800;
set output {frame_path};
set multiplot;

set size 0.4, 0.3;
set origin 0.55, 0.0;
set xrange [0:{timestep}] noextend;
set yrange [*:*];
set xlabel 'Timestep';
set ylabel 'Total game score';
set margins 5,0,4,0;
{rl_score_subplot}

set size 0.4, 0.3;
set origin 0.55, 0.3;
set yrange [150:450];
unset xtics;
unset xlabel;
set ylabel 'Cell tx power';
set margins 5,0,2,2;
{cell_config_subplot}

set xrange [*:*] noextend;
set yrange [0:50000];
set y2range [0:2];
unset ytics;
unset title;
set key off;
unset margin;
{ue_rx_subplots}

set size 0.5,0.95;
set origin 0.0,0.0;
set size square;
set xrange [*:*] noextend;
set yrange [*:*] noextend;
set xtics autofreq;
set ytics autofreq;
unset colorbox;
unset ylabel;
set label at screen 0.25,0.96 center "{{/:Bold FORLORN}}";
set label at screen 0.75,0.96 center "{subtitle}";
{main_plot}

unset multiplot;
reset;
'''

MAP_ONLY_TEMPLATE = '''
set terminal pdfcairo size 3.5,3.5;

set size square;
set xrange [*:*] noextend;
set yrange [*:*] noextend;
unset xtics;
unset ytics;
unset colorbox;
set label "{{/:Bold=30 {label}}}" at screen 0.08,0.85 left front tc rgb "white";
{main_plot}

reset;
'''

CONFIG_ONLY_TEMPLATE = '''
set terminal pdfcairo size 1.5,1.0;

set yrange [200:400];
unset xtics;
unset ytics; #set ytics ("20" 200, "30" 300, "40" 400);
{cell_config_subplot}

reset;
'''

TRIAL_TEMPLATE = '''
set terminal pdfcairo size 6,4;
set output {frame_path};
set multiplot;

set size 1, 0.5;
set origin 0.0, 0.5;
set xrange [0:{timestep}] noextend;
unset xtics;
set ylabel 'Total game score';
set margins 10,4,5,2;
{rl_score_subplot}

set origin 0.0, 0.0;
set yrange [200:400];
set ytics ("20 dBm" 200, "30 dBm" 300, "40 dBm" 400);
set xtics;
set xlabel 'Timestep';
set ylabel 'Cell tx power';
set margins 10,4,6,1;
{cell_config_subplot}

unset multiplot;
reset;
'''

CELL_OBJECT_TEMPLATE = 'polygon from %f,%f to %f,%f to %f,%f to %f,%f front lw 2'
CELL_LABEL_TEMPLATE = '"%d" at %f,%f center front'
UE_OBJECT_TEMPLATE_BG = 'circle center %f,%f size 35 front fs solid fc rgb "white"'
UE_OBJECT_TEMPLATE_FG = 'circle center %f,%f size 35 front lw 2 fc rgb "black"'
UE_LABEL_TEMPLATE = '"%d" at %f,%f center front'
ASSOC_ARROW_TEMPLATE = 'from %f,%f length %f angle %f front lw 4 lc rgb "#ffffff" nohead'

def offset_vector(x, y, direction, length):
    x2 = x + length * math.cos(direction * math.pi / 180)
    y2 = y + length * math.sin(direction * math.pi / 180)
    return (x2, y2)

class SimulationRenderer:
    def __init__(self, output_log, baseline_log=None, results_table=None, template=None, label=''):
        self.plot = GnuplotInterface()
        self.simulator = simulator.SimulatorLogReplay(output_log)
        self.results_table = results_table
        self.template = {
            None: FRAME_TEMPLATE,
            'map': MAP_ONLY_TEMPLATE,
            'config': CONFIG_ONLY_TEMPLATE,
            'trial': TRIAL_TEMPLATE,
        }[template]
        self.label = label
        if baseline_log is not None:
            self.parse_baseline_log(baseline_log)

    def parse_baseline_log(self, baseline_log):
        baseline_sim = simulator.SimulatorLogReplay(baseline_log)
        def save_baseline_score():
            baseline_score = baseline_sim.calculate_score()
            baseline_sim.add_data(RUN, 0, score=baseline_score)
        baseline_sim.run_parse_loop(periodic_callback=save_baseline_score)
        save_baseline_score()
        self.simulator.data[RUN][0] = {'baseline': baseline_sim.data[RUN][0]['score']}

    def get_ue_rx_subplot(self, ue):
        setup_commands = 'set ylabel "User %d" offset 1,0;' % ue
        self.plot.add_plot_element("'-' with filledcurves above y=0 notitle",
            self.simulator.get_timeseries(UE, ue, 'bytes_rx')[-100:])
        self.plot.add_plot_element("'-' with lines axes x1y2 lw 2 lc rgb '#336699' notitle",
            self.simulator.get_timeseries(UE, ue, 'score')[-100:])
        return setup_commands + self.plot.finish_plot_command()

    def get_ue_rx_subplots(self):
        items = list(self.simulator.get_objects(UE))
        area_w, area_h = 0.5, 0.35
        area_left, area_bottom = 0.5, 0.58
        layouts = []
        for cols in range(1, 20):
            rows = math.ceil(len(items) / cols)
            subplot_size = min(area_w / cols, area_h / rows) * 1.2
            layouts.append((subplot_size * subplot_size, rows, cols, subplot_size))
        utilization, rows, cols, subplot_size = max(layouts)
        col_w, row_h = area_w / cols, area_h / rows
        output = 'set size %f,%f;\n' % (subplot_size, subplot_size)
        for i, ue in enumerate(items):
            row, col = i // cols, i % cols
            output += 'set origin %f,%f;\n' % (
                area_left + col * col_w + (col_w - subplot_size) / 2,
                area_bottom + (rows - 1 - row) * row_h + (row_h - subplot_size) / 2)
            output += self.get_ue_rx_subplot(ue)
        return output

    def get_rl_score_subplot(self):
        if self.simulator.timeseries_valid(RUN, 0, 'baseline_none') \
                and self.simulator.timeseries_valid(RUN, 0, 'baseline_optuna'):
            self.plot.add_plot_element("'-' using 1:2:4 with filledcurves fc rgb '#ddeeff' notitle",
                zip(*zip(*self.simulator.get_timeseries(RUN, 0, 'baseline_none')),
                    *zip(*self.simulator.get_timeseries(RUN, 0, 'baseline_optuna'))))
            self.plot.add_plot_element("'-' with lines lw 4 lc rgb '#996666' notitle",
                self.simulator.get_timeseries(RUN, 0, 'baseline_none'))
            self.plot.add_plot_element("'-' with lines lw 4 lc rgb '#339966' notitle",
                self.simulator.get_timeseries(RUN, 0, 'baseline_optuna'))
        self.plot.add_plot_element("'-' with lines lw 2 lc rgb '#003366' notitle",
            self.simulator.get_timeseries(RUN, 0, 'score'))
        return self.plot.finish_plot_command()

    def get_cell_config_subplot(self):
        for cell in self.simulator.get_objects(CELL):
            self.plot.add_plot_element("'-' with steps lw 2 notitle",
                self.simulator.get_timeseries(CELL, cell, 'tx_power'))
        return self.plot.finish_plot_command()

    def get_map_path(self, map_config):
        map_path = 'output/map-%s.dat' % '-'.join(map(str, map_config))
        # Check that the map file exists, or run the map generator if not
        if not os.path.isfile(map_path):
            subprocess.run([ './simulator',
                    '--left=%s' % str(map_config[0]),
                    '--right=%s' % str(map_config[1]),
                    '--top=%s' % str(map_config[2]),
                    '--map=%s' % map_path ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        return map_path

    def get_plot_commands(self, frame_path=''):
        # Finish generating all the peripheral subplots first, before assembling
        # the main plot, because our GnuplotInterface object is stateful
        ue_rx_subplots = self.get_ue_rx_subplots()
        rl_score_subplot = self.get_rl_score_subplot()
        cell_config_subplot = self.get_cell_config_subplot()

        # Use REM heatplotted as the background of the canvas
        map_config = [ self.simulator.get_value(CELL, cell, 'tx_power') for cell in range(1, 4) ]
        self.plot.add_plot_element("'%s' using 1:2:(log10($4)) with image notitle" % self.get_map_path(map_config))

        # Render cells as triangles pointing in the antenna direction
        for cell in self.simulator.get_objects(CELL):
            x, y = self.simulator.get_value(CELL, cell, 'coords')
            direction = self.simulator.get_value(CELL, cell, 'direction')
            x2, y2 = offset_vector(x, y, direction - 40, 100)
            x3, y3 = offset_vector(x, y, direction + 40, 100)
            self.plot.add_object(CELL_OBJECT_TEMPLATE, x, y, x2, y2, x3, y3, x, y)
            lx, ly = offset_vector(x, y, direction, 50)
            self.plot.add_label(CELL_LABEL_TEMPLATE, cell, lx, ly)

        # Render UEs as filled circles
        for ue in self.simulator.get_objects(UE):
            x, y = self.simulator.get_value(UE, ue, 'coords')
            self.plot.add_object(UE_OBJECT_TEMPLATE_BG, x, y)
            self.plot.add_object(UE_OBJECT_TEMPLATE_FG, x, y)
            self.plot.add_label(UE_LABEL_TEMPLATE, ue, x, y)

            # If the UE is connected/associated to a cell, draw a line segment from
            # the tip of the cell's triangle to the edge of the UE's circle
            cur_associated_cell = self.simulator.get_value(UE, ue, 'cell_associated')
            if cur_associated_cell is not None:
                cx, cy = self.simulator.get_value(CELL, cur_associated_cell, 'coords')
                direction = self.simulator.get_value(CELL, cur_associated_cell, 'direction')
                cx, cy = offset_vector(cx, cy, direction, 100 * math.cos(40 * math.pi / 180))
                angle = math.atan2(y - cy, x - cx) * 180 / math.pi
                length = math.sqrt((cx - x) ** 2 + (cy - y) ** 2) - 25
                self.plot.add_arrow(ASSOC_ARROW_TEMPLATE, cx, cy, length, angle)

        # Put together the plot subtitle string
        subtitle = 'Level %s, timestep %d, config %d %d %d, score %.2f' % (
            self.simulator.current_seed,
            self.simulator.timestep,
            self.simulator.get_value(CELL, 1, 'tx_power'),
            self.simulator.get_value(CELL, 2, 'tx_power'),
            self.simulator.get_value(CELL, 3, 'tx_power'),
            self.simulator.get_value(RUN, 0, 'score'))

        # Use the frame template to assemble the final plot script
        return self.template.format(
            frame_path=frame_path,
            ue_rx_subplots=ue_rx_subplots,
            rl_score_subplot=rl_score_subplot,
            cell_config_subplot=cell_config_subplot,
            timestep=self.simulator.timestep,
            label=self.label,
            subtitle=subtitle,
            main_plot=self.plot.finish_plot_command(),
        )

    def save_current_scores(self):
        # Save the current trial score
        current_trial_score = self.simulator.calculate_score()
        self.simulator.add_data(RUN, 0, score=current_trial_score)
        # Also store each user's individual experience score at this timestep
        for ue in self.simulator.get_objects(UE):
            self.simulator.add_data(UE, ue, score=self.simulator.calculate_score_for_ue(ue))
        # If supplied with a results table on the command line, fetch and save
        # the relevant baseline results for the current timestep
        if self.results_table is not None:
            baseline_none, baseline_optuna = float('nan'), float('nan')
            key = 'Level %s' % self.simulator.current_seed
            if key in self.results_table:
                baseline_none = self.results_table[key]['baseline']
                baseline_optuna = self.results_table[key]['optuna']
            self.simulator.add_data(RUN, 0, baseline_none=baseline_none, baseline_optuna=baseline_optuna)

    @staticmethod
    def render_animation(output_log, baseline_log=None, results_table=None,
                current_frame=0, start_time=None, end_time=None, frame_skip=1, template=None):
        renderer = SimulationRenderer(output_log, baseline_log, results_table, template=template)
        renderer.current_frame = current_frame
        renderer.n = 0
        def render_frame():
            renderer.save_current_scores()
            renderer.n += 1
            if (renderer.n - 1) % frame_skip != 0:
                return
            if start_time is not None and renderer.simulator.timestep < start_time:
                return
            if end_time is not None and renderer.simulator.timestep > end_time:
                return
            frame_path = "'output/frame%d.png'" % renderer.current_frame
            plot_commands = renderer.get_plot_commands(frame_path)
            renderer.plot.run_commands(plot_commands)
            renderer.current_frame += 1
            sys.stdout.write('.')
            sys.stdout.flush()
        renderer.simulator.run_parse_loop(periodic_callback=render_frame)
        render_frame()
        print('')
        return renderer.current_frame

    @staticmethod
    def render_final_frame(output_log, template=None, label=''):
        renderer = SimulationRenderer(output_log, template=template, label=label)
        renderer.simulator.run_parse_loop(periodic_callback=renderer.save_current_scores)
        renderer.save_current_scores()
        plot_commands = renderer.get_plot_commands()
        return GnuplotInterface.render_single_frame(plot_commands)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+', help='The log files for all the runs to be included in the rendering')
    parser.add_argument('--baseline', help='Log file containing a baseline trial to compare against')
    parser.add_argument('--results', help='JSON file containing a results table to compare against')
    parser.add_argument('--start-time', type=int, default=0, help='Simulation timestep to start rendering at')
    parser.add_argument('--end-time', type=int, default=None, help='Simulation timestep to end rendering at')
    parser.add_argument('--frame-skip', type=int, default=1, help='How many frames to skip between each rendered frame')
    parser.add_argument('--template', default=None, help='A specialized plot template to use')
    args = parser.parse_args()

    baseline_log = None
    if args.baseline is not None:
        assert (len(args.logs) == 1)
        with open(args.baseline, 'r') as f:
            baseline_log = f.readlines()

    results_table = None
    if args.results is not None:
        with open(args.results, 'r') as f:
            results_table = json.load(f)

    current_frame = 0
    for log in args.logs:
        sys.stdout.write(log)
        sys.stdout.flush()
        with open(log, 'r') as f:
            current_frame = SimulationRenderer.render_animation(
                output_log=f.readlines(),
                baseline_log=baseline_log,
                results_table=results_table,
                template=args.template,
                current_frame=current_frame,
                start_time=args.start_time,
                end_time=args.end_time,
                frame_skip=args.frame_skip)
