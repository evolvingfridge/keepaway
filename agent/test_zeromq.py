# import zmq

# context = zmq.Context()

# subscriber = context.socket (zmq.SUB)
# subscriber.connect ("tcp://192.168.55.112:5556")
# subscriber.setsockopt (zmq.SUBSCRIBE, "NASDAQ")

# publisher = context.socket (zmq.PUB)
# publisher.bind ("ipc://nasdaq-feed")

# while True:
#     message = subscriber.recv()
#     publisher.send (message)


#
#   Hello World client in Python
#   Connects REQ socket to tcp://localhost:5555
#   Sends "Hello" to server, expects "World" back
#
import random
import zmq
import time
from keepaway_pb2 import StepIn, StepOut
from dql.dql_agent import DQLAgent

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5558")

stepIn = StepIn()
stepOut = StepOut()

agents = [DQLAgent(), DQLAgent(), DQLAgent()]
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
    # elif not stepIn.state:
    #     print('\n')
    #     print('endEpisode')
    #     print('=' * 20)
    #     print('\n\n\n\n')
    # time.sleep(1)
    # stepOut.action = random.randint(1, 3)
    stepOut.action = action
    out = stepOut.SerializeToString()
    # socket.send(b"Hello")
    print('Sending')
    socket.send(out)
    print('Send')
