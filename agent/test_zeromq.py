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
from keepaway_pb2 import StepIn, StepOut

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket = context.socket(zmq.REP)
socket.connect("tcp://localhost:5555")

stepIn = StepIn()
stepOut = StepOut()

while True:
    message = socket.recv()
    stepIn.ParseFromString(message)
    print("Received [ reward={}, step={} ]".format(stepIn.reward, stepIn.state))

    stepOut.action = random.randint(1, 10)
    out = stepOut.SerializeToString()
    # socket.send(b"Hello")
    print('Sending')
    socket.send(out)
