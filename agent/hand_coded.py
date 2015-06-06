import logging
import numpy as np

logger = logging.getLogger('keepaway')


class HandCodedAgent(object):
    def __init__(self, **kwargs):
        self._episode_started = True
        self.epsilon = 1

    def _get_network_dump(self):
        return ''

    def start_episode(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, current_state, *args, **kwargs):
        numKeepers = 3
        numTakers = 2

        state = current_state

        j = 0
        WB_dist_to_C = state[j]
        j += 1

        WB_dist_to_K = [-1]
        for i in range(1, numKeepers):
            WB_dist_to_K.append(state[j])
            j += 1

        WB_dist_to_T = []
        for i in range(0, numTakers):
            WB_dist_to_T.append(state[j])
            j += 1

        dist_to_C_K = [-1]
        for i in range(1, numKeepers):
            dist_to_C_K.append(state[j])
            j += 1

        dist_to_C_T = []
        for i in range(0, numTakers):
            dist_to_C_T.append(state[j])
            j += 1

        nearest_Opp_dist_K = [-1]
        for i in range(1, numKeepers):
            nearest_Opp_dist_K.append(state[j])
            j += 1

        nearest_Opp_ang_K = [-1]
        for i in range(1, numKeepers):
            nearest_Opp_ang_K.append(state[j])
            j += 1

        if WB_dist_to_T[0] > 5:
            return 0

        scores = [90]
        dist_weight = 4.0

        for i in range(1, numKeepers):
            scores.append(dist_weight * nearest_Opp_dist_K[i] + nearest_Opp_ang_K[i])

        return np.argmax(scores)

    def end_episode(self, current_time, *args, **kwargs):
        pass
