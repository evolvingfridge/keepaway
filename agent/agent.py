import argparse
import cPickle
import datetime
import logging
import os
import uuid

import zmq
from keepaway_pb2 import StepIn, StepOut

from dql.dql_agent import DQLAgent
# from hand_coded import HandCodedAgent


class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=False, default=None, **kwargs):
        if envvar and envvar in os.environ:
            default = os.environ[envvar]
            if kwargs.get('nargs') in ('*', '+'):
                default = map(lambda x: kwargs['type'](x.strip()), default.split(','))
            else:
                type_ = kwargs.get('type', str)
                if type_ == bool:
                    default = (default.lower() in ('1', 'true'))
        if required and default:
            required = False
        super(EnvDefault, self).__init__(default=default, required=required,
                                         **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)

parser = argparse.ArgumentParser(description='DQL Agent.')

# DQL params
parser.add_argument('--network-architecture', metavar='N', type=int, nargs='*',
                    help='Deep Network architecture', action=EnvDefault, envvar='NETWORK_ARCHITECTURE')
parser.add_argument('--train', dest='train', action='store_true',
                    default=True, help='Train network?')

parser.add_argument('--minibatch-size', type=int, action=EnvDefault, envvar='MINIBATCH_SIZE')
parser.add_argument('--transitions-history-size', type=int, action=EnvDefault, envvar='TRANSITIONS_HISTORY_SIZE')
parser.add_argument('--recent-states-to-network', type=int, action=EnvDefault, envvar='RECENT_STATES_TO_NETWORK')
parser.add_argument('--train-batch', type=bool, action=EnvDefault, envvar='TRAIN_BATCH')

parser.add_argument('--discount-factor', type=float, action=EnvDefault, envvar='DISCOUNT_FACTOR')
parser.add_argument('--final-epsilon-greedy', type=float, action=EnvDefault, envvar='FINAL_EPSILON_GREEDY')
parser.add_argument('--start-learning-rate', type=float, action=EnvDefault, envvar='START_LEARNING_RATE')
parser.add_argument('--final-learning-rate', type=float, action=EnvDefault, envvar='FINAL_LEARNING_RATE')
parser.add_argument('--learning-rate-change-episodes', type=float, action=EnvDefault, envvar='LEARNING_RATE_CHANGE_EPISODES')
parser.add_argument('--constant-learning-rate', type=float, action=EnvDefault, envvar='CONSTANT_LEARNING_RATE')
# parser.add_argument('--learning-rate', type=float, action=EnvDefault, envvar='LEARNING_RATE')
parser.add_argument('--rmsprop-rho', type=float, action=EnvDefault, envvar='RMSPROP_RHO')

parser.add_argument('--start-learn-after', type=int, action=EnvDefault, envvar='START_LEARN_AFTER')
parser.add_argument('--exploration-time', type=float, action=EnvDefault, envvar='EXPLORATION_TIME')
parser.add_argument('--stop-after-episodes', type=int, action=EnvDefault, envvar='STOP_AFTER_EPISODES')

parser.add_argument('--use-rmsprop', type=bool, action=EnvDefault, envvar='USE_RMSPROP')
parser.add_argument('--update-rule', type=str, action=EnvDefault, envvar='UPDATE_RULE')
parser.add_argument('--error-func', type=str, default='mean', metavar='E', choices=['sum', 'mean'], action=EnvDefault, envvar='ERROR_FUNC')
parser.add_argument('--swap-networks-every', type=int, action=EnvDefault, envvar='SWAP_NETWORKS_EVERY')

parser.add_argument('--clip-delta', type=float, action=EnvDefault, envvar='CLIP_DELTA')

parser.add_argument('--use-lasagne', type=bool, action=EnvDefault, envvar='USE_LASAGNE')

parser.add_argument('--multi-agent', type=bool, default=True, action=EnvDefault, envvar='MULTI_AGENT')
parser.add_argument('--keepers-count', type=int, default=3, action=EnvDefault, envvar='KEEPERS_COUNT')

# other params
parser.add_argument('--evaluate-agent-each', type=int, default=600,  metavar='X', help='Evaluate network (without training) every X episodes', action=EnvDefault, envvar='EVALUATE_AGENT_EACH')
parser.add_argument('--evaluation-episodes', type=int, default=100,  metavar='Y', help='Evaluation time (in episodes)', action=EnvDefault, envvar='EVALUATION_EPISODES')
parser.add_argument('--evaluation-epsilon', type=float, default=0.0, action=EnvDefault, envvar='EVALUATION_EPSILON')
parser.add_argument('--final-evaluation', type=int, default=1000, action=EnvDefault, envvar='FINAL_EVALUATION')

parser.add_argument('--logger-level', type=str, default='WARNING',  metavar='L', choices=['WARNING', 'INFO', 'DEBUG'], help='Logger level', action=EnvDefault, envvar='LOGGER_LEVEL')
parser.add_argument('--load-nnets', type=str, action=EnvDefault, envvar='LOAD_NNETS')
parser.add_argument('--initial-episodes', type=int, default=25000, action=EnvDefault, envvar='INITIAL_EPISODES')

args = parser.parse_args()

file_params = (
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    str(uuid.uuid4()).split('-')[0]
)

# logging
logging_level = getattr(logging, args.logger_level)
logger = logging.getLogger('keepaway')
logger.setLevel(logging_level)
# create file handler which logs even debug messages

fh = logging.FileHandler(os.path.expanduser('~/logs/agent-{}-{}.log'.format(*file_params)))
fh.setLevel(logging_level)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging_level)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s %(message)s', "%Y-%m-%d %H:%M:%S")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# save networks file
network_filepath = os.path.expanduser('~/logs/networks-{}-{}.log'.format(*file_params))


def main():
    context = zmq.Context()
    logger.info("Starting keepaway agent server...")
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5558")

    stepIn = StepIn()
    stepOut = StepOut()

    agent_kwargs = {k: v for (k, v) in args._get_kwargs() if v is not None}
    for v in ('learning_rate', 'constant_learning_rate'):
        if agent_kwargs.get(v):
            agent_kwargs['start_learning_rate'] = agent_kwargs['final_learning_rate'] = agent_kwargs[v]

    episodes_count = 1
    regular_episodes = 1
    evaluation_episodes_count = 0
    evaluation = False
    episode_started = False
    in_final_eval = False

    agents = [DQLAgent(**agent_kwargs) for i in range(args.keepers_count if args.multi_agent else 1)]
    if args.load_nnets:
        print('using defined nnets')
        for i, a in enumerate(agents):
            with open(os.path.expanduser('~/logs/' + args.load_nnets + '__agent_{}'.format(i)), 'r') as f:
                print(f.name)
                a.nnet = cPickle.load(f)
                a.episodes_played = args.initial_episodes
                a.nnet.episodes_played = args.initial_episodes
                a._epsilon = args.final_epsilon_greedy
                a.learning_rate = args.final_learning_rate
        episodes_count = args.initial_episodes
        regular_episodes = args.initial_episodes
    # agent = DQLAgent(**agent_kwargs)
    # agent = HandCodedAgent(**agent_kwargs)
    pid2id = {}
    current_id = 0

    logger.info('Ready to receive...')
    while True:
        logger.debug('=' * 40)
        message = socket.recv()
        stepIn.ParseFromString(message)
        logger.debug("Received [ reward={}, state={}, pid={}, end={}, current_time={} ]".format(
            stepIn.reward, stepIn.state, stepIn.player_pid, stepIn.episode_end, stepIn.current_time
        ))

        if stepIn.player_pid not in pid2id:
            pid2id[stepIn.player_pid] = current_id
            if args.multi_agent:
                current_id += 1

        agent = agents[pid2id[stepIn.player_pid]]

        # start episode
        if stepIn.reward == -1:
            episode_started = True
            action = agent.start_episode(
                current_time=stepIn.current_time,
                current_state=stepIn.state
            )
            logger.warning('start episode {} for {}'.format(episodes_count, stepIn.player_pid))
        elif stepIn.episode_end:
            if episode_started:
                episodes_count += 1
                if not evaluation:
                    regular_episodes += 1
                if episodes_count % 100 == 1 and not evaluation:
                    logger.warning('Episodes: {}...; current epsilon: {}; current learning rate: {}; frames played: {}'.format(
                        ', '.join(map(str, [episodes_count - 1, regular_episodes] + [a.episodes_played for a in agents])),
                        ', '.join(map(str, [a.epsilon for a in agents])),
                        ', '.join(map(str, [a.learning_rate for a in agents])),
                        ', '.join(map(str, [a.memory.entries_count for a in agents])),
                    ))
            agent.end_episode(current_time=stepIn.current_time)
            logger.warning('end episode {} for {}'.format(episodes_count-1, stepIn.player_pid))
            for a in agents:
                a.episodes_played = regular_episodes
            if args.stop_after_episodes:
                if episodes_count >= (args.stop_after_episodes - args.final_evaluation):
                    in_final_eval = True
                if args.stop_after_episodes == episodes_count:
                    break
            action = 0

            # evaluation
            if episode_started:
                if evaluation or in_final_eval:
                    evaluation_episodes_count += 1
                if episodes_count > 0 and (episodes_count % args.evaluate_agent_each == 1 or in_final_eval):
                    for a in agents:
                        a.train = False
                    if not evaluation:
                        logger.warning('Starting evaluation at {} (simulator time: {}) with epsilon {}'.format(episodes_count, stepIn.current_time, agent.epsilon))
                    evaluation = True
                if evaluation_episodes_count == args.evaluation_episodes and not in_final_eval:
                    logger.warning('Evaluation end after {} episodes (simulator time: {}, total episodes: {})'.format(evaluation_episodes_count, stepIn.current_time, episodes_count - 1))
                    evaluation = False
                    for a in agents:
                        a.train = True
                    evaluation_episodes_count = 0
                    # save current network
                    # with open(network_filepath + '__eval_{}'.format(episodes_count), 'a') as f:
                    #     cPickle.dump(agent.nnet, f, -1)

            episode_started = False
        else:
            action = agent.step(
                current_time=stepIn.current_time,
                current_state=stepIn.state
            )
        stepOut.action = action
        out = stepOut.SerializeToString()
        socket.send(out)
    for i, agent in enumerate(agents):
        with open(network_filepath + '__agent_{}'.format(i), 'a') as f:
            cPickle.dump(agent.nnet, f, -1)

if __name__ == '__main__':
    main()
