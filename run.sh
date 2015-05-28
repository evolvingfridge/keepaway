#!/bin/bash
echo "building keepaway player"
cd /home/soccer/keepaway/player
make
echo "going to sleep"
sleep 30  # let theano process neural network
echo "starting keepaway in sync mode"
cd
./keepaway/keepaway.py --keeper-policy=dql --keeper-learn --keeper-output=logs/keeper.out --log-dir=logs/ --synch-mode

