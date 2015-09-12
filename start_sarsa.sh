#!/bin/bash
cd /home/soccer
./keepaway/keepaway.py --keeper-policy=learn --keeper-learn --keeper-output=logs/keeper.out --log-dir=logs/ --synch-mode && sleep ${@: -1}
