import argparse

import zmq
from keepaway_pb2 import StepIn, StepOut

from dql.dql_agent import DQLAgent

parser = argparse.ArgumentParser(description='DQL Agent.')
parser.add_argument('--network-architecture', metavar='N', type=int, nargs='+',
                    help='Deep Network architecture')
parser.add_argument('--train', dest='train', action='store_true',
                    default=True, help='Train network?')
parser.add_argument('--minibatch-size', type=int)
parser.add_argument('--transitions-history-size', type=int)
parser.add_argument('--recent-states-to-network', type=int)
parser.add_argument('--discount-factor', type=float)
parser.add_argument('--learning-rate', type=float)

args = parser.parse_args()


def main():
    context = zmq.Context()
    print("Starting keepaway agent server...")
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5558")

    stepIn = StepIn()
    stepOut = StepOut()

    agent_kwargs = {k: v for (k, v) in args._get_kwargs() if v is not None}

    agents = [DQLAgent(**agent_kwargs), DQLAgent(**agent_kwargs), DQLAgent(**agent_kwargs)]
    pid2id = {}
    current_id = 0

    while True:
        print('Receiving')
        message = socket.recv()
        stepIn.ParseFromString(message)
        print("Received [ reward={}, state={}, pid={}, end={} ]".format(stepIn.reward, stepIn.state, stepIn.player_pid, stepIn.episode_end))

        if stepIn.player_pid not in pid2id:
            pid2id[stepIn.player_pid] = current_id
            current_id += 1

        agent = agents[pid2id[stepIn.player_pid]]

        # start episode
        if stepIn.reward == -1:
            print('\n')
            print('=' * 20)
            print('startEpisode')
            action = agent.start_episode(reward=0, current_state=stepIn.state)
        elif stepIn.episode_end:
            agent.end_episode(stepIn.reward)
            action = 0
        else:
            action = agent.step(reward=stepIn.reward, current_state=stepIn.state)
        stepOut.action = action
        out = stepOut.SerializeToString()
        print('Sending')
        socket.send(out)
        print('Send')


if __name__ == '__main__':
    main()
