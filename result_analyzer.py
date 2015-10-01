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
parser.add_argument('--draw-constants', action='store_true', default=True)
# parser.add_argument('--use-learning-time', action='store_true', default=True)
parser.add_argument('--evaluation-each', default=None, type=int)
parser.add_argument('--evaluation-length', default=None, type=int)
parser.add_argument('--window-write-each', default=10, type=int)
parser.add_argument('--window-mean-write-each', default=10, type=int)

args = parser.parse_args()

Q_VALUE_RE = re.compile(r'Q-Value \(episode: (\d+), step: ([\d.]+), action: (\d)\): ([\d.]+)')
ERROR_RE = re.compile(r'Error \(episode: (\d+), step: ([\d.]+)\): ([\d.]+)')
HISTOGRAM_MAX = 70

STATS = """
avg episode length (evaluated, {avg_episodes_eval_count}): {avg_episode_eval_length} (+- {avg_episode_eval_length_stdev})
avg episodes played: {avg_played}
avg total (simulator) time played [h]: {avg_time_played}
median episode length: {median_episode_length}

avg episode length (not evaluated, {avg_episodes_count}): {avg_episode_length} (+- {avg_episode_length_stdev})

=======
{agent_env}
"""

PLOT_EXT = 'svg'


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


def get_agent_env():
    agent_env_path = os.path.expanduser(os.path.join(args.logs_directory, 'agent.env'))
    if os.path.exists(agent_env_path):
        return '\n'.join(open(agent_env_path, 'r').readlines())
    return 'CONF NOT PROVIDED'


# http://code.activestate.com/recipes/511478/
def percentile(N, percent, key=lambda x: x):
    """
    Find the percentile of a list of values.

    @parameter N - is a list of values. Note N MUST BE already sorted.
    @parameter percent - a float value from 0.0 to 1.0.
    @parameter key - optional key function to compute value from each element of N.

    @return - the percentile of the values
    """
    if not N:
        return None
    N = sorted(N)
    k = (len(N)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(N[int(k)])
    d0 = key(N[int(f)]) * (c-k)
    d1 = key(N[int(c)]) * (k-f)
    return d0 + d1


def save_graph(additional_opts, series):
    options = {
        'min_y': "0",
        'min_x': "-0.05",
        'title': 'Graph',
        'series': series,
        'y_title': 'Episode Duration (seconds)',
        'terminal': 'svg',
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


def save_eval_graph(additional_opts):
    options = {
        'max_x': "10000",
        'max_y': "30",
        'title': 'Graph',
        'y_title': 'Episode Duration (seconds)',
        'terminal': 'svg',
    }
    window_opts = options.copy()
    window_opts.update(additional_opts)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'graph_eval.gnuplot.tmpl')) as graph_tmpl:
        with tempfile.NamedTemporaryFile('w') as f_window_graph:
            g = graph_tmpl.read().format(**window_opts)
            f_window_graph.write(g)
            f_window_graph.flush()
            print(f_window_graph.name)
            subprocess.call(['gnuplot', f_window_graph.name])


def save_histogram(additional_opts):
    options = {
        'max': int(additional_opts['max_episode_length']),
        'n': additional_opts['max_episode_length'] * 1,
        'title': 'Episode length histogram',
        'terminal': 'svg',
    }
    histogram_opts = options.copy()
    histogram_opts.update(additional_opts)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'histogram.gnuplot.tmpl')) as graph_tmpl:
        with tempfile.NamedTemporaryFile('w') as f_histogram:
            g = graph_tmpl.read().format(**histogram_opts)
            f_histogram.write(g)
            f_histogram.flush()
            print(f_histogram.name)
            subprocess.call(['gnuplot', f_histogram.name])


# def process_kwy(f_evaluation_std, f_evaluation_confidence, f_window, f_window_episodes, f_window_mean, f_window_episodes_mean):
def process_kwy(f_window, f_window_episodes, f_stats, f_histogram, f_evaluation_raw, f_evaluation_stats):
    evaluation_each, evaluation_length = get_evaluation_params()
    out_files = (f_window, f_window_episodes)
    constants_files = (f_window, f_window_episodes)
    i = 0
    max_episode_length = 0
    final_episodes_length = []
    final_episodes_eval_length = []
    episodes_counts = []
    keepaway_total_times = []
    evaluation_stats = {}
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
        episodes_count_without_eval = 0
        j = 0
        current_sum = 0
        hours = 0.0
        last_non_eval_hours = 0.0
        hours_diff = 0.0
        evaluation_episodes = []
        evaluations = 0
        eval_mean = 0
        print('Processing {}'.format(f_full))
        with open(f_full) as f_obj:
            for line in f_obj.readlines():
                if line.startswith('#'):
                    continue
                start_time, end_time, episode_length = map(int, line.split()[1:-1])
                episode_length /= 10.0
                episodes_count += 1
                hours = start_time / (10.0 * 3600)

                # window
                evaluation = episodes_count % evaluation_each <= evaluation_length
                if episodes_count < evaluation_each:
                    evaluation = False
                if not evaluation:
                    current_sum -= episodes_window[j]
                    episodes_window[j] = episode_length
                    current_sum += episode_length
                    j = (j + 1) % args.window_size
                    episodes_count_without_eval += 1
                    last_non_eval_hours = hours
                    if episodes_count >= args.window_size and episodes_count % args.window_write_each == 0:
                        f_window.write('\t'.join(map(str, (
                            hours - hours_diff,
                            current_sum / args.window_size,
                            '\n'
                        ))))
                        f_window_episodes.write('\t'.join(map(str, (
                            episodes_count_without_eval,
                            current_sum / args.window_size,
                            '\n'
                        ))))

                # evaluation
                else:
                    if episodes_count % evaluation_each < evaluation_length:
                        evaluation_episodes.append(episode_length)
                    elif episodes_count % evaluation_each == evaluation_length:
                        evaluations += 1
                        hours_diff += (hours - last_non_eval_hours)
                        if evaluation_episodes:
                            eval_mean = statistics.mean(evaluation_episodes)
                            # stdev = statistics.stdev(evaluation_episodes, xbar=eval_mean)
                            # confidence = 1.96 * stdev / math.sqrt(len(evaluation_episodes))
                            # f_evaluation_raw.write(' '.join(map(str, (
                            #     episodes_count,
                            #     eval_mean,
                            #     stdev,
                            #     '\n',
                            # ))))
                            # f_evaluation_confidence.write(' '.join(map(str, (
                            #     episodes_count,
                            #     eval_mean,
                            #     confidence,
                            #     '\n',
                            # ))))
                            evaluation_episodes = []
                            evaluation_stats.setdefault(episodes_count, []).append(eval_mean)
                # window mean
                # if episodes_count >= args.window_size and episodes_count % args.window_mean_write_each == 0:
                #     median = statistics.median(episodes_window)
                #     f_window_mean.write('\t'.join(map(str, (
                #         hours,
                #         median,
                #         '\n'
                #     ))))
                #     f_window_episodes_mean.write('\t'.join(map(str, (
                #         episodes_count,
                #         median,
                #         '\n'
                #     ))))

                # evaluation
            final_episodes_eval_length.append(eval_mean)
            final_episodes_length.append(current_sum / args.window_size)
            # final_episodes_length.append(statistics.mean(episodes_window))
            episodes_counts.append(episodes_count_without_eval)
            keepaway_total_times.append(hours - hours_diff)
        for out_f in out_files:
            out_f.write('\n\n')

        for val in episodes_window:
            if val <= HISTOGRAM_MAX:
                f_histogram.write(str(val) + '\n')
                max_episode_length = max(max_episode_length, int(val))

    # f_evaluation_raw.write('Deep-Q-Learning\n')
    # f_evaluation_stats.write('Deep-Q-Learning\n')
    f_evaluation_stats.write('\t'.join((
        '# x',
        'min',
        '1_quart',
        'avg',
        'median',
        'conf',
        'stdev',
        '3_quart',
        'max',
        '\n'
    )))
    max_eval = 0
    for i, (c, values) in enumerate(sorted(evaluation_stats.items(), key=lambda x: x[0])):
        eval_mean = statistics.mean(values)
        eval_median = statistics.median(values)
        stdev = statistics.stdev(values, xbar=eval_mean) if len(values) > 1 else 0
        confidence = 1.96 * stdev / math.sqrt(len(values))
        first_quartile = percentile(values, 0.25)
        third_quartile = percentile(values, 0.75)
        # f_evaluation_raw.write(' '.join(map(str, (
        #     c,
        #     eval_mean,
        #     stdev,
        #     '\n',
        # ))))
        max_eval = c - (i+2) * evaluation_length
        f_evaluation_stats.write('\t'.join(map(str, (
            max_eval,                        # 1
            round(min(values), 2),    # 2
            round(first_quartile, 2), # 3
            round(eval_mean, 2),      # 4
            round(eval_median, 2),    # 5
            round(confidence, 2),     # 6
            round(stdev, 2),          # 7
            round(third_quartile, 2), # 8
            round(max(values), 2),    # 9
            '\n',
        ))))

        for val in values:
            f_evaluation_raw.write(' '.join(map(str, (max_eval, val, '\n'))))

    # boxplot
    # f_evaluation_raw.write('# ')
    # for key in sorted(evaluation_stats.keys()):
    #     f_evaluation_raw.write('{} '.format(key))
    # f_evaluation_raw.write('\n')
    # for c in range(len(list(evaluation_stats.values())[0])):
    #     for key in sorted(evaluation_stats.keys()):
    #         f_evaluation_raw.write('{} '.format(evaluation_stats[key][c]))
    #     f_evaluation_raw.write('\n')

    f_evaluation_raw.write('\n\n')
    f_evaluation_stats.write('\n\n')

    if args.draw_constants:
        for out_f in constants_files:
            for metric, val, stdev in (('random', 5.3, 1.8), ('always-hold', 2.9, 1.0), ('hand-coded', 13.3, 8.3)):
                out_f.write('{}\n'.format(metric))
                for tick in (0, (hours - hours_diff) if out_f in (f_window,) else episodes_count_without_eval):
                    out_f.write(' '.join(map(str, (tick, val, stdev, '\n'))))
                out_f.write('\n\n')
    for out_f in out_files + constants_files + (f_evaluation_raw, f_evaluation_stats):
        out_f.flush()

    series = i + (3 * args.draw_constants)

    f_stats.write(STATS.format(
        avg_episodes_eval_count=evaluation_length,
        avg_episode_eval_length=statistics.mean(final_episodes_eval_length),
        avg_episode_eval_length_stdev=statistics.stdev(final_episodes_eval_length) if len(final_episodes_length) > 1 else 0,

        avg_episodes_count=args.window_size,
        avg_episode_length=statistics.mean(final_episodes_length),
        avg_episode_length_stdev=statistics.stdev(final_episodes_length) if len(final_episodes_length) > 1 else 0,

        avg_played=statistics.mean(episodes_counts),
        avg_time_played=statistics.mean(keepaway_total_times),
        median_episode_length=statistics.median(final_episodes_length),
        agent_env=get_agent_env(),
    ))
    # window episodes
    save_graph({
        'cols': '1:2',
        'file': f_window.name,
        'out_file': os.path.join(args.logs_directory, 'window_graph.{}'.format(PLOT_EXT)),
        'plot_options': 'w lines',
        'x_title': 'Training Time (simulator hours)',
        'title': 'Avg episode duration (win size: {})'.format(args.window_size),
    }, series)

    save_graph({
        'cols': '1:2',
        'file': f_window_episodes.name,
        'out_file': os.path.join(args.logs_directory, 'window_graph_episodes.{}'.format(PLOT_EXT)),
        'plot_options': 'w lines',
        'x_title': 'Episodes count',
        'title': 'Avg episode duration (win size: {})'.format(args.window_size),
    }, series)

    save_histogram({
        'file': f_histogram.name,
        'max_episode_length': max_episode_length,
        'out_file': os.path.join(args.logs_directory, 'histogram_graph.{}'.format(PLOT_EXT)),
        'div': 1.0 / i,
    })

    # save_graph({
    #     'cols': '1:2',
    #     'file': f_window_mean.name,
    #     'out_file': os.path.join(args.logs_directory, 'window_graph_mean.{}'.format(PLOT_EXT)),
    #     'plot_options': 'w lines',
    #     'x_title': 'Training Time (simulator hours)',
    #     'title': 'Median episode duration (win size: {})'.format(args.window_size),
    # }, series)

    # save_graph({
    #     'cols': '1:2',
    #     'file': f_window_episodes_mean.name,
    #     'out_file': os.path.join(args.logs_directory, 'window_graph_episodes_mean.{}'.format(PLOT_EXT)),
    #     'plot_options': 'w lines',
    #     'x_title': 'Episodes count',
    #     'title': 'Median episode duration (win size: {})'.format(args.window_size),
    # }, series)

    save_graph({
        'cols': '1:4:7',
        'file': f_evaluation_stats.name,
        'out_file': os.path.join(args.logs_directory, 'window_graph_eval_std.{}'.format(PLOT_EXT)),
        'plot_options': 'w yerrorbars',
        'x_title': 'Episodes count',
        'title': 'Avg episode duration during evaluation with std ({} every {} episodes)'.format(evaluation_length, evaluation_each),
    }, series)

    save_graph({
        'cols': '1:4:6',
        'file': f_evaluation_stats.name,
        'out_file': os.path.join(args.logs_directory, 'window_graph_eval_conf.{}'.format(PLOT_EXT)),
        'plot_options': 'w yerrorbars',
        'x_title': 'Episodes count',
        'title': 'Avg episode duration during evaluation with confidence ({} every {} episodes)'.format(evaluation_length, evaluation_each),
    }, series)

    save_graph({
        'cols': '1:5',
        'file': f_evaluation_stats.name,
        'out_file': os.path.join(args.logs_directory, 'window_graph_eval_median.{}'.format(PLOT_EXT)),
        # 'plot_options': 'w yerrorbars',
        'plot_options': 'w points',
        'x_title': 'Episodes count',
        'title': 'Median episode duration during evaluation ({} every {} episodes)'.format(evaluation_length, evaluation_each),
    }, series)

    save_eval_graph({
        'file_stats': f_evaluation_stats.name,
        'file_raw_data': f_evaluation_raw.name,
        'point_offset': 0.01 * max(evaluation_stats.keys()),
        'max_x': max_eval + 500,
        'max_y': max([max(v) for v in evaluation_stats.values()]) * 1.2,
        'out_file': os.path.join(args.logs_directory, 'window_graph_eval_box.{}'.format(PLOT_EXT)),
        'x_title': 'Episodes count',
        'title': 'Average episode duration during evaluation ({} every {} episodes)'.format(evaluation_length, evaluation_each-evaluation_length),
    })


def process_agent_logs(f_mean_q_delta, f_mean_q_steps, f_mean_starting_q):
    """
    f_mean_q_delta - average delta in episode in time
        for single episode is AVG(|Q_predicted - Q_expected|) which is AVG(error)

    f_mean_q_steps - average Q (predicted) based on step number
    """
    evaluation_each, evaluation_length = get_evaluation_params()
    # TODO: respect eval
    out_files = (f_mean_q_delta, f_mean_q_steps, f_mean_starting_q)
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

            STARTING_Q_MEAN = 50
            starting_q = [0] * STARTING_Q_MEAN
            starting_q_current_i = 0

            for line in f_obj:
                q_val = re.search(Q_VALUE_RE, line)  # returns (episode, step, action, Q)
                error_val = re.search(ERROR_RE, line)  # returns (episode, step, error)

                if q_val:
                    episode, step, action, q = q_val.groups()
                    step = int(float(step))
                    steps_sum_q[step] += float(q)
                    steps_count_q[step] += 1
                    if step == 0:
                        starting_q[starting_q_current_i] = float(q)
                        starting_q_current_i = (starting_q_current_i + 1) % STARTING_Q_MEAN
                        mean = statistics.mean(starting_q)
                        f_mean_starting_q.write(' '.join(map(str, (episode, mean, '\n'))))
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
            for step in range(1, max(steps_sum_q.keys() or [1]) + 1):
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
        'out_file': os.path.join(args.logs_directory, 'mean_q_delta.{}'.format(PLOT_EXT)),
        'plot_options': 'w lines',
        'x_title': 'Episodes',
        'y_title': '|Q_expected - Q_predicted|',
        'title': 'Avg Q delta (network error) in episode in time',
    }, series)

    save_graph({
        'cols': '1:3',
        'file': f_mean_q_steps.name,
        'out_file': os.path.join(args.logs_directory, 'mean_q_steps.{}'.format(PLOT_EXT)),
        'plot_options': 'w lines',
        'x_title': 'Steps',
        'y_title': 'Avg Q',
        'title': 'Avg Q (predicted) based on step number',
    }, series)

    save_graph({
        'cols': '1:2',
        'file': f_mean_starting_q.name,
        'out_file': os.path.join(args.logs_directory, 'mean_starting_q_steps.{}'.format(PLOT_EXT)),
        'plot_options': 'w lines',
        'x_title': 'Steps',
        'y_title': 'Avg Q for step 0',
        'title': 'Avg Q (predicted) in step 0',
    }, series)


def main():
    if not os.path.isdir(args.logs_directory):
        print("logs directory isn't directory")
        sys.exit(1)
    if args.logs_directory.endswith('/'):
        args.logs_directory = args.logs_directory[:-1]

    # process kwy files
    with tempfile.NamedTemporaryFile('w') as f_window:
        with tempfile.NamedTemporaryFile('w') as f_window_episodes:
            with open(os.path.join(args.logs_directory, 'stats.txt'), 'w') as f_stats:
                with tempfile.NamedTemporaryFile('w') as f_histogram:
                    with tempfile.NamedTemporaryFile('w') as f_evaluation_std:
                        with tempfile.NamedTemporaryFile('w') as f_evaluation_confidence:
                            #     with tempfile.NamedTemporaryFile('w') as f_window_episodes_mean:
                            process_kwy(f_window, f_window_episodes, f_stats, f_histogram, f_evaluation_std, f_evaluation_confidence)

    # process agent log files
    # with tempfile.NamedTemporaryFile('w') as f_mean_q_delta:
    #     with tempfile.NamedTemporaryFile('w') as f_mean_q_steps:
    #         with tempfile.NamedTemporaryFile('w') as f_mean_starting_q:
    #             process_agent_logs(f_mean_q_delta, f_mean_q_steps, f_mean_starting_q)

if __name__ == '__main__':
    main()
