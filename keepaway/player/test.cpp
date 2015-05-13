#include <string.h>  // for memcpy
#include <stdlib.h>  // for rand
#include <unistd.h>
#include <iostream>
#include "zmq.hpp"
#include "keepaway.pb.h"

#ifndef LEARNING_AGENT
#define LEARNING_AGENT

class LearningAgent
{
    zmq::context_t* zmq_context;
    zmq::socket_t* zmq_socket;
    int selectAction( double reward, double state[], bool wait4action );

    public:
        LearningAgent();
        ~LearningAgent();

        int  startEpisode( double state[] );
        int  step( double reward, double state[] );
        void endEpisode( double reward );
        void setParams(int iCutoffEpisodes, int iStopLearningEpisodes){;};
} ;

#endif

/* Constructor:
   numFeatures and numActions are required.  Additional parameter
   can be added to the constructor.  For example:
*/
LearningAgent::LearningAgent()
{
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
    std::cout << "selecting action" << std::endl;
    keepaway::StepIn stepIn;
    stepIn.set_reward(reward);
    if(wait4action){
        for ( int v = 0; v < 13/*getNumFeatures()*/; v++ ) {
            stepIn.add_state(state[v]);
        }
    }
    std::string buf;
    stepIn.SerializeToString(&buf);
    zmq::message_t request (buf.size());
    memcpy ((void *) request.data(), buf.c_str(), buf.size());
    zmq_socket->send(request);
    std::cout << "receiving action" << std::endl;

    int action = 0;
    keepaway::StepOut stepOut;
    zmq::message_t reply;
    zmq_socket->recv (&reply);
    std::string rpl = std::string(static_cast<char*>(reply.data()), reply.size());
    stepOut.ParseFromString(rpl);
    action = stepOut.action();
    std::cout << "received " << action << std::endl;
    return action;
}

int main(){
    LearningAgent *la = new LearningAgent();
    double state[]{1,2,3,4,5,6,7,8,9,10,11,12,13};
    la->startEpisode(state);
    la->step(10, state);
    la->step(20, state);
    la->endEpisode(5);
    std::cout << "end " << std::endl;
    return 0;
}
