#include <string.h>  // for memcpy
#include <stdlib.h>  // for rand
#include "LearningAgent.h"

/* Constructor:
   numFeatures and numActions are required.  Additional parameter
   can be added to the constructor.  For example:
*/
   LearningAgent::LearningAgent( int numFeatures, int numActions,
    bool learning,
    char *loadWeightsFile,
    char *saveWeightsFile ):
   SMDPAgent( numFeatures, numActions )
   {

  /* Contruct learner here */
  /* For example: */
    /*
    m_learning = learning;
    strcpy( m_saveWeightsFile, saveWeightsFile );
    m_saving = ( m_learning && strlen( saveWeightsFile ) > 0 );

    if ( strlen( loadWeightsFile ) > 0 ) {
      loadWeights( loadWeightsFile );
  }
  */

}

/* Start Episode
   Use the current state to choose an action [0..numKeepers-1]
*/
   int LearningAgent::startEpisode( double state[] )
   {
  /* Choose first action */
  /* For example: */
    /*
    m_lastAction = selectAction( state );
    memcpy( m_lastState, state, getNumFeatures() * sizeof( double ) );
    return m_lastAction;
    */
}

/* Step
   Update based on previous step and choose next action
*/
   int LearningAgent::step( double reward, double state[] )
   {
    /*
    if ( m_learning )
      update( m_lastState, m_lastAction, reward );
  m_lastAction = selectAction( state );
  memcpy( m_lastState, state, getNumFeatures() * sizeof( double ) );
  return m_lastAction;
  */
  return ( rand() % getNumActions() );
}

void LearningAgent::endEpisode( double reward )
{
    /*
    if ( m_learning )
      update( m_lastState, m_lastAction, reward );
  if ( m_saving && rand() % 200 == 0 )
      saveWeights( m_saveWeightsFile );
      */
}

void LearningAgent::setParams(int iCutoffEpisodes, int iStopLearningEpisodes)
{
  /* set learning parameters */
}


/* Your private methods here.
   For example:
*/

   void LearningAgent::loadWeights ( char *filename )
   {
   }

   void LearningAgent::saveWeights ( char *filename )
   {
   }

   void LearningAgent::update( double state[], int action, double reward )
   {
   }

   int LearningAgent::selectAction( double state[] )
   {
  // Choose a random action
    return rand() % getNumActions();
}
