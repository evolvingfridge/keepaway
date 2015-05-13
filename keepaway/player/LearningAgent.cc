#include <string.h>  // for memcpy
#include <stdlib.h>  // for rand
#include <unistd.h>
#include "LearningAgent.h"
#include "zmq.hpp"
#include "keepaway.pb.h"

/* Constructor:
   numFeatures and numActions are required.  Additional parameter
   can be added to the constructor.  For example:
*/
LearningAgent::LearningAgent(
    int numFeatures,
    int numActions,
    bool learning,
    char *loadWeightsFile,
    char *saveWeightsFile
): SMDPAgent( numFeatures, numActions )
{
    //  Prepare our context and socket
    // zmq::context_t context (1);
    // zmq::socket_t socket (context, ZMQ_REP);
    zmq_context = new zmq::context_t(1);
    zmq_socket = new zmq::socket_t(*zmq_context, ZMQ_REQ);
    zmq_socket->bind ("tcp://localhost:5555");

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
    return selectAction( 0, state, true);
}

/* Step
   Update based on previous step and choose next action
*/
int LearningAgent::step( double reward, double state[] )
{
    return selectAction( reward, state, true);
}

void LearningAgent::endEpisode( double reward )
{
    double state[0];
    selectAction(reward, state, false);
}

int LearningAgent::selectAction( double reward, double state[], bool wait4action)
{
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
    memcpy ((void *) request.data(), buf.c_str(), buf.size());
    zmq_socket->send(request);

    int action = 0;
    return action;
    if(wait4action){
        keepaway::StepOut stepOut;
        zmq::message_t reply;
        zmq_socket->recv (&reply);
        std::string rpl = std::string(static_cast<char*>(reply.data()), reply.size());
        stepOut.ParseFromString(rpl);
        action = stepOut.action();
    }
    return action;
}