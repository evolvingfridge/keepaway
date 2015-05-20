/*
Copyright (c) 2004 Gregory Kuhlmann, Peter Stone
University of Texas at Austin
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the University of Amsterdam nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <string.h>  // for memcpy
#include <stdlib.h>  // for rand
#include <unistd.h>
#include "zmq.hpp"
#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <iostream>

// #include "LearningAgent.h"
#include "keepaway.pb.h"
#include "HandCodedAgent.h"

using namespace std;

HandCodedAgent::HandCodedAgent( int numFeatures, int numActions,
				char *strPolicy, WorldModel *wm ):
  SMDPAgent( numFeatures, numActions )
{
    strcpy( policy, strPolicy );
    WM = wm;
    std::cout << "[" << ::getpid() <<  "] hand coded init" << std::endl;
    zmq_context = new zmq::context_t(1);
    zmq_socket = new zmq::socket_t(*zmq_context, ZMQ_REQ);
    zmq_socket->connect("tcp://localhost:5558");

    GOOGLE_PROTOBUF_VERIFY_VERSION;
}

HandCodedAgent::~HandCodedAgent()
{
    delete zmq_context;
    delete zmq_socket;
}

int HandCodedAgent::startEpisode( double state[] )
{
    // std::cout << "[" << ::getpid() <<  "] start episode" << std::endl;
    return step( -1 , state );
}

int HandCodedAgent::step( double reward, double state[] )
{
    // std::cout << "[" << ::getpid() <<  "] step" << std::endl;
    return getAction(reward, state, false, getNumFeatures());
}

void HandCodedAgent::endEpisode( double reward )
{
    // std::cout << "[" << ::getpid() <<  "] end episode" << std::endl;
    // Do nothing
    double state[0];
    // for ( int v = 0; v < getNumFeatures(); v++ ) {
    //   state[v] = -1;
    // }
    getAction(reward, state, true, 0);
}

/************ POLICIES **************/

int HandCodedAgent::getAction( double reward, double state[], bool end, int features)
{
    // std::cout << "[" << ::getpid() <<  "] select action" << std::endl;

    // send reward and state
    keepaway::StepIn stepIn;
    stepIn.set_current_time(WM->getCurrentCycle());
    stepIn.set_reward(reward);
    for ( int v = 0; v < features; v++ ) {
        stepIn.add_state(state[v]);
    }
    stepIn.set_episode_end(end);
    stepIn.set_player_pid(::getpid());
    std::string buf;
    stepIn.SerializeToString(&buf);
    zmq::message_t request (buf.size());
    // std::cout << "[" << ::getpid() <<  "] sending" << std::endl;
    memcpy ((void *) request.data(), buf.c_str(), buf.size());
    zmq_socket->send(request);
    // std::cout << "[" << ::getpid() <<  "] send" << std::endl;

    // receive action
    int action = 0;
    keepaway::StepOut stepOut;
    zmq::message_t reply;
    // std::cout << "[" << ::getpid() <<  "] receiving" << std::endl;
    zmq_socket->recv (&reply);
    // std::cout << "[" << ::getpid() <<  "] received" << std::endl;
    std::string rpl = std::string(static_cast<char*>(reply.data()), reply.size());
    stepOut.ParseFromString(rpl);
    action = stepOut.action();
    // std::cout << "[" << ::getpid() <<  "] end " << action << std::endl;
    return action;
}
