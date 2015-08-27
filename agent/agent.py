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
parser.add_argument('--discount-factor', type=float, action=EnvDefault, envvar='DISCOUNT_FACTOR')
# parser.add_argument('--learning-rate', type=float, action=EnvDefault, envvar='LEARNING_RATE')
parser.add_argument('--start-learn-after', type=int, action=EnvDefault, envvar='START_LEARN_AFTER')
parser.add_argument('--evaluation-epsilon', type=int, action=EnvDefault, envvar='EVALUATION_EPSILON')
parser.add_argument('--exploration-time', type=float, action=EnvDefault, envvar='EXPLORATION_TIME')
parser.add_argument('--train-batch', type=bool, action=EnvDefault, envvar='TRAIN_BATCH')
parser.add_argument('--use-rmsprop', type=bool, action=EnvDefault, envvar='USE_RMSPROP')
parser.add_argument('--update-rule', type=str, action=EnvDefault, envvar='UPDATE_RULE')
parser.add_argument('--error-func', type=str, default='mean', metavar='E', choices=['sum', 'mean'], action=EnvDefault, envvar='ERROR_FUNC')
parser.add_argument('--final-epsilon-greedy', type=float, action=EnvDefault, envvar='FINAL_EPSILON_GREEDY')
parser.add_argument('--rmsprop-rho', type=float, action=EnvDefault, envvar='RMSPROP_RHO')

parser.add_argument('--start-learning-rate', type=float, action=EnvDefault, envvar='START_LEARNING_RATE')
parser.add_argument('--final-learning-rate', type=float, action=EnvDefault, envvar='FINAL_LEARNING_RATE')
parser.add_argument('--learning-rate-change-episodes', type=float, action=EnvDefault, envvar='LEARNING_RATE_CHANGE_EPISODES')
parser.add_argument('--constant-learning-rate', type=float, action=EnvDefault, envvar='CONSTANT_LEARNING_RATE')
parser.add_argument('--clip-delta', type=float, action=EnvDefault, envvar='CLIP_DELTA')

parser.add_argument('--use-lasagne', type=bool, action=EnvDefault, envvar='USE_LASAGNE')
parser.add_argument('--stop-after-episodes', type=int, action=EnvDefault, envvar='STOP_AFTER_EPISODES')
parser.add_argument('--swap-networks-every', type=int, action=EnvDefault, envvar='SWAP_NETWORKS_EVERY')

# other params
parser.add_argument('--evaluate-agent-each', type=int, default=5000,  metavar='X', help='Evaluate network (without training) every X episodes', action=EnvDefault, envvar='EVALUATE_AGENT_EACH')
parser.add_argument('--evaluation-episodes', type=int, default=200,  metavar='Y', help='Evaluation time (in episodes)', action=EnvDefault, envvar='EVALUATION_EPISODES')
parser.add_argument('--logger-level', type=str, default='WARNING',  metavar='L', choices=['WARNING', 'INFO', 'DEBUG'], help='Logger level', action=EnvDefault, envvar='LOGGER_LEVEL')

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
    agent = DQLAgent(**agent_kwargs)
    # agent = HandCodedAgent(**agent_kwargs)
    pid2id = {}
    current_id = 0

    episodes_count = 0
    evaluation_episodes_count = 0
    evaluation = False

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
            current_id += 1

        # start episode
        if stepIn.reward == -1:
            action = agent.start_episode(
                current_time=stepIn.current_time,
                current_state=stepIn.state
            )
        elif stepIn.episode_end:
            if agent._episode_started:
                episodes_count += 1
                if episodes_count % 100 == 0:
                    logger.warning('Episodes: {}...; current epsilon: {}; current learning rate: {}; frames played: {}'.format(
                        episodes_count,
                        agent.epsilon,
                        agent.learning_rate,
                        agent.memory.entries_count,
                    ))
            agent.end_episode(current_time=stepIn.current_time)
            if args.stop_after_episodes and args.stop_after_episodes == episodes_count:
                with open(network_filepath, 'a') as f:
                    cPickle.dump(agent.nnet, f, -1)
                break
            action = 0

            # evaluation
            if evaluation:
                evaluation_episodes_count += 1
            if episodes_count > 0 and episodes_count % args.evaluate_agent_each == 0:
                logger.debug('Starting evaluation at {}'.format(episodes_count))
                evaluation = True
                agent.train = False
            if evaluation_episodes_count == args.evaluation_episodes:
                logger.debug('Evaluation end at {} (total: {})'.format(evaluation_episodes_count, episodes_count))
                evaluation = False
                agent.train = True
                evaluation_episodes_count = 0
                # save current network
                with open(network_filepath, 'a') as f:
                    f.write(agent._get_network_dump())
                    f.write('\n\n')
        else:
            action = agent.step(
                current_time=stepIn.current_time,
                current_state=stepIn.state
            )
        stepOut.action = action
        out = stepOut.SerializeToString()
        socket.send(out)
        # logger.debug('=' * 40)


if __name__ == '__main__':
    main()
