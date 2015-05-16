#include <string.h>  // for memcpy
#include <stdlib.h>  // for rand
#include <unistd.h>
#include "LearningAgent.h"
#include "zmq.hpp"
#include "keepaway.pb.h"
#include <iostream>
#include <fstream>

using namespace std;

// #include "Logger.h"

// extern Logger Log;

/* Constructor:
   numFeatures and numActions are required.  Additional parameter
   can be added to the constructor.  For example:
*/
LearningAgent::LearningAgent(
    int numFeatures, int numActions
): SMDPAgent( numFeatures, numActions )
{
    std::cout << "[" << ::getpid() <<  "] init" << std::endl;
    //  Prepare our context and socket
    // zmq::context_t context (1);
    // zmq::socket_t socket (context, ZMQ_REP);
    zmq_context = new zmq::context_t(1);
    zmq_socket = new zmq::socket_t(*zmq_context, ZMQ_REQ);
    zmq_socket->bind ("tcp://*:5555");

    GOOGLE_PROTOBUF_VERIFY_VERSION;
}

LearningAgent::~LearningAgent()
{
    delete zmq_context;
    delete zmq_socket;
}

/* Start Episode
   Use the current state to choose an action [0..numKeepers-1]
*/
int LearningAgent::startEpisode( double state[] )
{
    std::cout << "[" << ::getpid() <<  "] start episode" << std::endl;
    return selectAction( 0, state, true);
}

/* Step
   Update based on previous step and choose next action
*/
int LearningAgent::step( double reward, double state[] )
{
    std::cout << "[" << ::getpid() <<  "] step" << std::endl;
    return selectAction( reward, state, true);
}

void LearningAgent::endEpisode( double reward )
{
    std::cout << "[" << ::getpid() <<  "] end episode" << std::endl;
    double state[0];
    selectAction(reward, state, false);
}

int LearningAgent::selectAction( double reward, double state[], bool wait4action)
{
    /*
    std::cout << "[" << ::getpid() <<  "] select action" << std::endl;
    keepaway::StepIn stepIn;
    stepIn.set_reward(reward);
    if(wait4action){
        for ( int v = 0; v < getNumFeatures(); v++ ) {
            stepIn.add_state(state[v]);
        }
    }
    std::string buf;
    stepIn.SerializeToString(&buf);
    zmq::message_t request (buf.size());
    std::cout << "[" << ::getpid() <<  "] sending" << std::endl;
    memcpy ((void *) request.data(), buf.c_str(), buf.size());
    zmq_socket->send(request);
    std::cout << "[" << ::getpid() <<  "] send" << std::endl;

    int action = 0;
    // return action;
    // if(wait4action){
        keepaway::StepOut stepOut;
        zmq::message_t reply;
        std::cout << "[" << ::getpid() <<  "] receiving" << std::endl;
        zmq_socket->recv (&reply);
        std::cout << "[" << ::getpid() <<  "] received" << std::endl;
        std::string rpl = std::string(static_cast<char*>(reply.data()), reply.size());
        stepOut.ParseFromString(rpl);
        action = stepOut.action();
    // }
    std::cout << "[" << ::getpid() <<  "] end " << action << std::endl;
    return action;
    */
    return 0;
}
