import argparse
import sys
import os
import numpy as np


class Encoder():
    def __init__(self, parameters_path, statefile_path, q) -> None:
        self.action_dict = {'0': 0, '1': 1, '2': 2, '4': 3, '6': 4}
        self.outcome_dict = {'-1': 0, '0': 1,
                             '1': 2, '2': 3, '3': 4, '4': 5, '6': 6}
        self.num_states = 452
        self.num_actions = 5
        self.file = open("mdpfile.txt", "w")
        self.q = q
        self.parameters_path = parameters_path
        self.state_path = statefile_path
        self.action_prob_matrix = np.zeros((self.num_actions, 7))
        self.tail = False
        with open(self.parameters_path, 'r') as file:
            lines = file.readlines()[1:]
            for line in lines:
                action = line.split()[0]
                self.action_prob_matrix[self.action_dict[str(line[0])]] = (line.split()[
                    1:])

    def write_info(self):
        print("numStates " + str(self.num_states))
        print("numActions " + str(self.num_actions))
        print("end 451,452")

    def get_action_key(self, action):
        # This function provides the index in the array and return the action key
        return (list(self.action_dict.keys())
                [list(self.action_dict.values()).index(action)])

    def get_outcome_key(self, outcome):
        # This function provides the index in the array and return the outcome key
        return (list(self.outcome_dict.keys())
                [list(self.outcome_dict.values()).index(outcome)])


def parameters_path(string):
    if os.path.isfile(string):
        # print("MDP Path used %s" % string)
        return string
    else:
        raise NotADirectoryError(string)


def state_path(string):
    if os.path.isfile(string):
        # print("MDP Path used %s" % string)
        return string
    else:
        raise NotADirectoryError(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=state_path, default=None)
    parser.add_argument("--parameters", type=parameters_path, default=None)
    parser.add_argument("--q", type=float, default=0.25)

    args = parser.parse_args()

    encoder = Encoder("data\cricket\sample-p1.txt",
                      "data\cricket\cricket_state_list.txt", 0.25)
    encoder.write_info()
