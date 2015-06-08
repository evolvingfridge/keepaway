#!/usr/bin/env python
import argparse
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict

parser = argparse.ArgumentParser(description='Keepaway results analyzer.')
parser.add_argument('logs_directory', metavar='L', help='Logs directory')
# parser.add_argument('-g', '--graph', metavar='G', help='File with output gnuplot config')
parser.add_argument('--window-size', default=1000, type=int)
parser.add_argument('--mean-q-window-size', default=100, type=int)
parser.add_argument('--draw-constants', action='store_true', default=False)
# parser.add_argument('--use-learning-time', action='store_true', default=True)
parser.add_argument('--evaluation-each', default=None, type=int)
parser.add_argument('--evaluation-length', default=None, type=int)
parser.add_argument('--window-write-each', default=1, type=int)
parser.add_argument('--window-mean-write-each', default=1, type=int)

args = parser.parse_args()

Q_VALUE_RE = re.compile(r'Q-Value \(episode: (\d+), step: ([\d.]+), action: (\d)\): ([\d.]+)')
ERROR_RE = re.compile(r'Error \(episode: (\d+), step: ([\d.]+)\): ([\d.]+)')


def get_evaluation_params():
    evaluation_each = 2000
    evaluation_length = 100

    # try to read some values from agent.env
    agent_env_path = os.path.join(args.logs_directory, 'agent.env')
    if os.path.exists(agent_env_path):
        for line in open(agent_env_path, 'r').readlines():
            if not line.strip():
                continue
            var_name, value = line.split('=', 1)
            if var_name == 'EVALUATE_AGENT_EACH':
                evaluation_each = int(value)
            if var_name == 'EVALUATION_EPISODES':
                evaluation_length = int(value)

    evaluation_each = args.evaluation_each or evaluation_each
    evaluation_length = args.evaluation_length or evaluation_length
    return evaluation_each, evaluation_length


def save_graph(additional_opts, series):
    options = {
        'min_y': "0",
        'min_x': "-0.05",
        'title': 'Graph',
        'series': series,
        'y_title': 'Episode Duration (seconds)',
    }
    window_opts = options.copy()
    window_opts.update(additional_opts)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'graph.gnuplot.tmpl')) as graph_tmpl:
        with tempfile.NamedTemporaryFile('w') as f_window_graph:
            g = graph_tmpl.read().format(**window_opts)
            f_window_graph.write(g)
            f_window_graph.flush()
            print(f_window_graph.name)
            subprocess.call(['gnuplot', f_window_graph.name])
            # import ipdb; ipdb.set_trace()


def process_kwy(f_evaluation_std, f_evaluation_confidence, f_window, f_window_episodes, f_window_mean, f_window_episodes_mean):
    evaluation_each, evaluation_length = get_evaluation_params()
    out_files = (f_evaluation_std, f_evaluation_confidence, f_window, f_window_episodes, f_window_mean, f_window_episodes_mean)
    i = 0
    for f in os.listdir(args.logs_directory):
        f_name, f_ext = os.path.splitext(f)
        f_full = os.path.join(args.logs_directory, f)
        if f_ext not in ('.kwy', ):
            continue
        i += 1
        for out_f in out_files:
            out_f.write('"{}"\n'.format(f_name.replace('_', ' ')))
        episodes_window = [0] * args.window_size
        episodes_count = 0
        j = 0
        current_sum = 0
        hours = 0.0
        evaluation_episodes = []
        evaluations = 0
        print('Processing {}'.format(f_full))
        with open(f_full) as f_obj:
            for line in f_obj.readlines():
                if line.startswith('#'):
                    continue
                start_time, end_time, episode_length = map(int, line.split()[1:-1])
                episode_length /= 10.0
                episodes_count += 1

                # window
                current_sum -= episodes_window[j]
                episodes_window[j] = episode_length
                current_sum += episode_length
                j = (j + 1) % args.window_size
                hours = start_time / (10.0 * 3600)
                if episodes_count >= args.window_size and episodes_count % args.window_write_each == 0:
                    f_window.write('\t'.join(map(str, (
                        hours,
                        current_sum / args.window_size,
                        '\n'
                    ))))
                    f_window_episodes.write('\t'.join(map(str, (
                        episodes_count,
                        current_sum / args.window_size,
                        '\n'
                    ))))

                # window mean
                if episodes_count >= args.window_size and episodes_count % args.window_mean_write_each == 0:
                    median = statistics.median(episodes_window)
                    f_window_mean.write('\t'.join(map(str, (
                        hours,
                        median,
                        '\n'
                    ))))
                    f_window_episodes_mean.write('\t'.join(map(str, (
                        episodes_count,
                        median,
                        '\n'
                    ))))

                # evaluation
                if episodes_count % evaluation_each < evaluation_length:
                    evaluation_episodes.append(episode_length)
                elif episodes_count % evaluation_each == evaluation_length:
                    evaluations += 1
                    mean = statistics.mean(evaluation_episodes)
                    stdev = statistics.stdev(evaluation_episodes, xbar=mean)
                    confidence = 1.96 * stdev / math.sqrt(len(evaluation_episodes))
                    f_evaluation_std.write(' '.join(map(str, (
                        episodes_count,
                        mean,
                        stdev,
                        '\n',
                    ))))
                    f_evaluation_confidence.write(' '.join(map(str, (
                        episodes_count,
                        mean,
                        confidence,
                        '\n',
                    ))))
                    evaluation_episodes = []
        for out_f in out_files:
            out_f.write('\n\n')

    if args.draw_constants:
        for out_f in out_files:
            for metric, val, stdev in (('random', 5.3, 1.8), ('always-hold', 2.9, 1.0), ('hand-coded', 13.3, 8.3)):
                out_f.write('{}\n'.format(metric))
                for tick in (0, hours if out_f in (f_window_mean, f_window) else episodes_count):
                    out_f.write(' '.join(map(str, (tick, val, stdev, '\n'))))
                out_f.write('\n\n')
    for out_f in out_files:
        out_f.flush()

    series = i + (3 * args.draw_constants)
    # window episodes
    save_graph({
        'cols': '1:2',
        'file': f_window.name,
        'out_file': os.path.join(args.logs_directory, 'window_graph.eps'),
        'plot_options': 'w lines',
        'x_title': 'Training Time (simulator hours)',
        'title': 'Avg episode duration (win size: {})'.format(args.window_size),
    }, series)

    save_graph({
        'cols': '1:2',
        'file': f_window_episodes.name,
        'out_file': os.path.join(args.logs_directory, 'window_graph_episodes.eps'),
        'plot_options': 'w lines',
        'x_title': 'Episodes count',
        'title': 'Avg episode duration (win size: {})'.format(args.window_size),
    }, series)

    save_graph({
        'cols': '1:2',
        'file': f_window_mean.name,
        'out_file': os.path.join(args.logs_directory, 'window_graph_mean.eps'),
        'plot_options': 'w lines',
        'x_title': 'Training Time (simulator hours)',
        'title': 'Median episode duration (win size: {})'.format(args.window_size),
    }, series)

    save_graph({
        'cols': '1:2',
        'file': f_window_episodes_mean.name,
        'out_file': os.path.join(args.logs_directory, 'window_graph_episodes_mean.eps'),
        'plot_options': 'w lines',
        'x_title': 'Episodes count',
        'title': 'Median episode duration (win size: {})'.format(args.window_size),
    }, series)

    save_graph({
        'cols': '1:2:3',
        'file': f_evaluation_std.name,
        'out_file': os.path.join(args.logs_directory, 'window_graph_eval_std.eps'),
        'plot_options': 'w yerrorbars',
        'x_title': 'Episodes count',
        'title': 'Avg episode duration during evaluation with std ({} every {} episodes)'.format(evaluation_length, evaluation_each),
    }, series)

    save_graph({
        'cols': '1:2:3',
        'file': f_evaluation_confidence.name,
        'out_file': os.path.join(args.logs_directory, 'window_graph_eval_conf.eps'),
        'plot_options': 'w yerrorbars',
        'x_title': 'Episodes count',
        'title': 'Avg episode duration during evaluation with confidence ({} every {} episodes)'.format(evaluation_length, evaluation_each),
    }, series)


def process_agent_logs(f_mean_q_delta, f_mean_q_steps):
    """
    f_mean_q_delta - average delta in episode in time
        for single episode is AVG(|Q_predicted - Q_expected|) which is AVG(error)

    f_mean_q_steps - average Q (predicted) based on step number
    """
    out_files = (f_mean_q_delta, f_mean_q_steps)
    i = 0
    for f in os.listdir(args.logs_directory):
        f_name, f_ext = os.path.splitext(f)
        f_full = os.path.join(args.logs_directory, f)
        if not f_name.startswith('agent') or f == 'agent.env':
            continue
        i += 1
        for out_f in out_files:
            # out_f.write('"{}"\n'.format(f_name.split('_')[-1].replace('_', ' ')))
            out_f.write('"{}"\n'.format(f.split('_')[-1].replace('_', ' ')))

        with open(f_full) as f_obj:
            current_episode = -1
            current_episode_error_sum = 0
            current_episode_actions_count = 0

            steps_sum_q = defaultdict(float)
            steps_count_q = defaultdict(int)

            for line in f_obj:
                q_val = re.search(Q_VALUE_RE, line)  # returns (episode, step, action, Q)
                error_val = re.search(ERROR_RE, line)  # returns (episode, step, error)

                if q_val:
                    episode, step, action, q = q_val.groups()
                    step = int(float(step))
                    steps_sum_q[step] += float(q)
                    steps_count_q[step] += 1
                if error_val:
                    episode, step, error = error_val.groups()
                    episode = int(episode)
                    step = int(float(step))
                    error = float(error)

                    current_episode_actions_count += 1
                    current_episode_error_sum += error

                    if episode > current_episode + args.mean_q_window_size:
                        if current_episode > 0:
                            f_mean_q_delta.write(' '.join(map(str, (
                                current_episode,
                                current_episode_error_sum,
                                current_episode_actions_count,
                                current_episode_error_sum / current_episode_actions_count if current_episode_actions_count != 0 else 0,
                                '\n'
                            ))))
                        current_episode = episode
                        current_episode_actions_count = 0
                        current_episode_error_sum = 0

            s = c = 0
            prev = 0
            for step in range(1, max(steps_sum_q.keys()) + 1):
                s = steps_sum_q.get(step, 0)
                c = steps_count_q.get(step, 0)
                if c != 0:
                    prev = s / c if c != 0 else 0
                f_mean_q_steps.write(' '.join(map(str, (
                    step,
                    c,
                    prev,
                    '\n'
                ))))
        for out_f in out_files:
            out_f.write('\n\n')
    for out_f in out_files:
        out_f.flush()

    series = i
    if not series:
        return
    save_graph({
        'cols': '1:4',
        'file': f_mean_q_delta.name,
        'out_file': os.path.join(args.logs_directory, 'mean_q_delta.eps'),
        'plot_options': 'w lines',
        'x_title': 'Episodes',
        'y_title': '|Q_expected - Q_predicted|',
        'title': 'Avg Q delta (network error) in episode in time',
    }, series)

    save_graph({
        'cols': '1:3',
        'file': f_mean_q_steps.name,
        'out_file': os.path.join(args.logs_directory, 'mean_q_steps.eps'),
        'plot_options': 'w lines',
        'x_title': 'Steps',
        'y_title': 'Avg Q',
        'title': 'Avg Q (predicted) based on step number',
    }, series)


def main():
    if not os.path.isdir(args.logs_directory):
        print("logs directory isn't directory")
        sys.exit(1)
    if args.logs_directory.endswith('/'):
        args.logs_directory = args.logs_directory[:-1]

    # process kwy files
    with tempfile.NamedTemporaryFile('w') as f_evaluation_std:
        with tempfile.NamedTemporaryFile('w') as f_evaluation_confidence:
            with tempfile.NamedTemporaryFile('w') as f_window:
                with tempfile.NamedTemporaryFile('w') as f_window_episodes:
                    with tempfile.NamedTemporaryFile('w') as f_window_mean:
                        with tempfile.NamedTemporaryFile('w') as f_window_episodes_mean:
                            process_kwy(f_evaluation_std, f_evaluation_confidence, f_window, f_window_episodes, f_window_mean, f_window_episodes_mean)

    # process agent log files
    with tempfile.NamedTemporaryFile('w') as f_mean_q_delta:
        with tempfile.NamedTemporaryFile('w') as f_mean_q_steps:
            process_agent_logs(f_mean_q_delta, f_mean_q_steps)

if __name__ == '__main__':
    main()
