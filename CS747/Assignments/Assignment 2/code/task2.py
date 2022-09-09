"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

You need to complete the following methods:
    - give_pull(self): This method is called when the algorithm needs to
        select the arms to pull for the next round. The method should return
        two arrays: the first array should contain the indices of the arms
        that need to be pulled, and the second array should contain how many
        times each arm needs to be pulled. For example, if the method returns
        ([0, 1], [2, 3]), then the first arm should be pulled 2 times, and the
        second arm should be pulled 3 times. Note that the sum of values in
        the second array should be equal to the batch size of the bandit.

    - get_reward(self, arm_rewards): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the rewards that were received. arm_rewards is a dictionary
        from arm_indices to a list of rewards received. For example, if the
        give_pull method returned ([0, 1], [2, 3]), then arm_rewards will be
        {0: [r1, r2], 1: [r3, r4, r5]}. (r1 to r5 are each either 0 or 1.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need.


def partition(batch_size):
    first_elem = int(0.4*batch_size)
    second_elem = int(0.25*batch_size)
    third_elem = int(0.15*batch_size)
    fourth_elem = int(0.1*batch_size)
    fifth_elem = batch_size - \
        (first_elem + second_elem + third_elem + fourth_elem)
    return np.array([first_elem, second_elem, third_elem, fourth_elem, fifth_elem], dtype=np.int64)

# END EDITING HERE


class AlgorithmBatched:
    def __init__(self, num_arms, horizon, batch_size):
        self.num_arms = num_arms
        self.horizon = horizon
        self.batch_size = batch_size
        assert self.horizon % self.batch_size == 0, "Horizon must be a multiple of batch size"
        # START EDITING HERE
        self.time = 2
        self.values = np.zeros(num_arms)
        self.ucb = np.zeros(num_arms)
        self.counts = np.ones(num_arms)
        # Add any other variables you need here
        # END EDITING HERE

    def give_pull(self):
        return np.argsort(self.ucb)[-5:], partition(self.batch_size)
        # END EDITING HERE

    def get_reward(self, arm_rewards):
        for arm_index in arm_rewards:
            self.counts[arm_index] += arm_rewards[arm_index].size
            self.time += self.batch_size
            self.values[arm_index] = (self.values[arm_index]*self.counts[arm_index]+np.sum(
                arm_rewards[arm_index]))/(self.counts[arm_index] + arm_rewards[arm_index].size)
        self.ucb = [min(self.values[i] + np.sqrt(
            (2*np.log(self.time)/self.counts[i])), 1) for i in range(self.num_arms)]
        # END EDITING HERE
