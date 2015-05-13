#ifndef LEARNING_AGENT
#define LEARNING_AGENT

#include "SMDPAgent.h"
#include "zmq.hpp"

class LearningAgent:public SMDPAgent
{
    zmq::context_t* zmq_context;
    zmq::socket_t* zmq_socket;
    int selectAction( double reward, double state[], bool wait4action );

    public:
        LearningAgent(
            int numFeatures,
            int numActions,
            bool learning,
            char *loadWeightsFile,
            char *saveWeightsFile
        );
        ~LearningAgent();

        int  startEpisode( double state[] );
        int  step( double reward, double state[] );
        void endEpisode( double reward );
        void setParams(int iCutoffEpisodes, int iStopLearningEpisodes){;};
} ;

#endif
