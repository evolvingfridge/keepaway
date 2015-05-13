./keepaway.py --keeper-policy=learning --keeper-learn --keeper-output=../logs/keeper.out --log-dir=../logs/

g++ -o test_zeromq.out test_zeromq.cpp -lzmq
g++-4.6 -o test_ipc test_ipc.cpp test.pb.cc -lprotobuf


# =============================================================================
protoc ~/keepaway/keepaway.proto --cpp_out=~/keepaway/player/ --python_out=~/agent/src/agent/
