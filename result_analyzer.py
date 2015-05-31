#!/usr/bin/env python
import argparse
import os
import statistics
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser(description='Keepaway results analyzer.')
parser.add_argument('logs_directory', metavar='L', help='Logs directory')
parser.add_argument('-o', '--output-graph', metavar='O', help='File with output graph')
parser.add_argument('-e', '--output-graph-eval', metavar='E', help='File with output evaluation graph')
# parser.add_argument('-g', '--graph', metavar='G', help='File with output gnuplot config')
parser.add_argument('--window-size', default=1000, type=int)
parser.add_argument('--use-epizodes-count', action='store_true', default=False)
parser.add_argument('--use-learning-time', action='store_true', default=True)
parser.add_argument('--evaluation-each', default=None, type=int)
parser.add_argument('--evaluation-length', default=None, type=int)


def main():
    args = parser.parse_args()
    if not os.path.isdir(args.logs_directory):
        print("logs directory isn't directory")
        sys.exit(1)
    if args.logs_directory.endswith('/'):
        args.logs_directory = args.logs_directory[:-1]

    evaluation_each = 2000
    evaluation_length = 100

    # try to read some values from agent.env
    agent_env_path = os.path.join(args.logs_directory, 'agent.env')
    if os.path.exists(agent_env_path):
        for line in open(agent_env_path, 'r').readlines():
            var_name, value = line.split('=', 1)
            if var_name == 'EVALUATE_AGENT_EACH':
                evaluation_each = int(value)
            if var_name == 'EVALUATION_EPISODES':
                evaluation_length = int(value)

    evaluation_each = args.evaluation_each or evaluation_each
    evaluation_length = args.evaluation_length or evaluation_length

    # temporary files to store window results
    with tempfile.NamedTemporaryFile('w') as f_evaluation:
        with tempfile.NamedTemporaryFile('w') as f_window:
            i = 0
            print('window file: ' + f_window.name)
            print('evaluation file: ' + f_evaluation.name)
            for f in os.listdir(args.logs_directory):
                f_name, f_ext = os.path.splitext(f)
                f_full = os.path.join(args.logs_directory, f)
                if f_ext not in ('.kwy'):
                    continue
                i += 1
                f_evaluation.write('"{}"\n'.format(f_name.replace('_', ' ')))
                f_window.write('"{}"\n'.format(f_name.replace('_', ' ')))
                episodes_window = [0] * args.window_size
                episodes_count = 0
                j = 0
                current_sum = 0
                hours = 0.0
                evaluation_episodes = []
                evaluations = 0
                for line in open(f_full).readlines():
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
                    if episodes_count >= args.window_size:
                        f_window.write('\t'.join(map(str, (
                            hours,
                            current_sum / args.window_size,
                            '\n'
                        ))))

                    # evaluation
                    if episodes_count % evaluation_each < evaluation_length:
                        evaluation_episodes.append(episode_length)
                    elif episodes_count % evaluation_each == evaluation_length:
                        evaluations += 1
                        mean = statistics.mean(evaluation_episodes)
                        f_evaluation.write(' '.join(map(str, (
                            episodes_count,
                            mean,
                            statistics.stdev(evaluation_episodes, xbar=mean),
                            '\n',
                        ))))
                        evaluation_episodes = []
                f_evaluation.write('\n\n')
                f_window.write('\n\n')
            f_evaluation.flush()
            f_window.flush()

            options = {
                'min_y': "0",
                'min_x': "-0.05",
                'title': 'Graph',
                'series': i,
            }
            # window graph
            window_opts = options.copy()
            window_opts.update({
                'cols': '1:2',
                'file': f_window.name,
                'out_file': args.output_graph or os.path.join(args.logs_directory, 'window_graph.eps'),
                'plot_options': 'w lines',
                'x_title': 'Training Time (simulator hours)',
            })
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'graph.gnuplot.tmpl')) as graph_tmpl:
                with tempfile.NamedTemporaryFile('w') as f_window_graph:
                    g = graph_tmpl.read().format(**window_opts)
                    f_window_graph.write(g)
                    f_window_graph.flush()
                    print(f_window_graph.name)
                    subprocess.call(['gnuplot', f_window_graph.name])
                    # import ipdb; ipdb.set_trace()

            # evaluation graph
            evaluation_opts = options.copy()
            evaluation_opts.update({
                'cols': '1:2:3',
                'file': f_evaluation.name,
                'out_file': args.output_graph_eval or os.path.join(args.logs_directory, 'window_graph_eval.eps'),
                'plot_options': 'w yerrorbars',
                'x_title': 'Episodes count'
            })
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tools', 'graph.gnuplot.tmpl')) as graph_tmpl:
                with tempfile.NamedTemporaryFile('w') as f_evaluation_graph:
                    g = graph_tmpl.read().format(**evaluation_opts)
                    f_evaluation_graph.write(g)
                    f_evaluation_graph.flush()
                    print(f_evaluation_graph.name)
                    subprocess.call(['gnuplot', f_evaluation_graph.name])
                    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
