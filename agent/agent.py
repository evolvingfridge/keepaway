import argparse
import datetime
import logging
import os
import uuid

import zmq
from keepaway_pb2 import StepIn, StepOut

from dql.dql_agent import DQLAgent


class EnvDefault(argparse.Action):
    def __init__(self, envvar, required=False, default=None, **kwargs):
        if not default and envvar:
            if envvar in os.environ:
                default = os.environ[envvar]
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
parser.add_argument('--learning-rate', type=float, action=EnvDefault, envvar='LEARNING_RATE')
parser.add_argument('--start-learn-after', type=int, action=EnvDefault, envvar='START_LEARN_AFTER')
parser.add_argument('--evaluation-epsilon', type=int, action=EnvDefault, envvar='EVALUATION_EPSILON')

# other params
parser.add_argument('--evaluate-agent-each', type=int, default=10000,  metavar='X', help='Evaluate network (without training) every X episodes', action=EnvDefault, envvar='EVALUATE_AGENT_EACH')
parser.add_argument('--evaluation-episodes', type=int, default=200,  metavar='Y', help='Evaluation time (in episodes)', action=EnvDefault, envvar='EVALUATION_EPISODES')

args = parser.parse_args()

file_params = (
    datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    str(uuid.uuid4()).split('-')[0]
)

# logging
logger = logging.getLogger('keepaway')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages

fh = logging.FileHandler(os.path.expanduser('~/logs/agent-{}-{}.log'.format(*file_params)))
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

# save networks file


def main():
    context = zmq.Context()
    logger.info("Starting keepaway agent server...")
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5558")

    stepIn = StepIn()
    stepOut = StepOut()

    agent_kwargs = {k: v for (k, v) in args._get_kwargs() if v is not None}

    agents = [DQLAgent(**agent_kwargs), DQLAgent(**agent_kwargs), DQLAgent(**agent_kwargs)]
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
        logger.debug("Received [ reward={}, state={}, pid={}, end={} ]".format(
            stepIn.reward, stepIn.state, stepIn.player_pid, stepIn.episode_end
        ))

        if stepIn.player_pid not in pid2id:
            pid2id[stepIn.player_pid] = current_id
            current_id += 1

        agent = agents[pid2id[stepIn.player_pid]]

        # start episode
        if stepIn.reward == -1:
            action = agent.start_episode(reward=0, current_state=stepIn.state)
        elif stepIn.episode_end:
            agent.end_episode(stepIn.reward)
            action = 0
            episodes_count += 1

            # evaluation
            if (episodes_count / 3) % args.evaluate_agent_each == 0:
                evaluation = True
                for agent in agents:
                    agent.train = False
            if evaluation:
                evaluation_episodes_count += 1
            if (evaluation_episodes_count / 3) % args.evaluation_episodes == 0:
                evaluation = False
                for agent in agents:
                    agent.train = True

        else:
            action = agent.step(reward=stepIn.reward, current_state=stepIn.state)
        stepOut.action = action
        out = stepOut.SerializeToString()
        socket.send(out)
        # logger.debug('=' * 40)


if __name__ == '__main__':
    main()
