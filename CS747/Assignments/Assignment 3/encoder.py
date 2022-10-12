import argparse
import sys
import os
import numpy as np


class Encoder():
    def __init__(self, statefile_path, parameters_path, q):

        self.q = q
        self.num_actions = 6

        self.state_path = statefile_path
        self.action_prob_matrix = np.zeros((self.num_actions, 7))
        self.tail = False
        self.action_matrix_dict = {}
        with open(parameters_path, 'r') as file:
            lines = file.readlines()[1:]

            for line in lines:
                self.action_matrix_dict[int(line.split()[0])] = [
                    float(prob) for prob in line.split()[1:]]

        # Creating the balls*runs*2 + 2 states using a dictionary mapping
        temp_states = []
        duplicate_states = []
        self.state_dict = {}

        with open(statefile_path, 'r') as file:
            lines = file.readlines()
            temp_states = [state.strip() for state in lines]

        for state in temp_states:
            duplicate_states.append(state + '0')
            duplicate_states.append(state + '1')

        # With all the states, we can now populate the state_dict
        self.state_dict = {state: index for index,
                           state in enumerate(duplicate_states)}

        self.max_balls = int(list(self.state_dict)[0][:2])
        self.max_runs = int(list(self.state_dict)[0][2:-1])
        self.num_states = self.max_balls*self.max_runs*2 + 2
        # Also need to define the end states in the state_dict

        # Win state is self.nums_states - 1

        for runs in range(1, self.max_runs+1):
            if runs < 10:
                self.state_dict["00" + "0" +
                                str(runs) + "0"] = self.num_states - 2
                self.state_dict["00" + "0" +
                                str(runs) + "1"] = self.num_states - 2
            else:
                self.state_dict["00" + str(runs) + "0"] = self.num_states - 2
                self.state_dict["00" + str(runs) + "1"] = self.num_states - 2

        for balls in range(self.max_balls+1):
            if balls < 10:
                self.state_dict["0" + str(balls) +
                                "00" + "0"] = self.num_states - 1
                self.state_dict["0" + str(balls) +
                                "00" + "1"] = self.num_states - 1
            else:
                self.state_dict[str(balls) + "00" + "0"] = self.num_states - 1
                self.state_dict[str(balls) + "00" + "1"] = self.num_states - 1

        self.available_actions = [0, 1, 2, 4, 6]
        self.action_dict = {action: index for index,
                            action in enumerate(self.available_actions)}
        # print(self.state_dict)
        self.tailender_actions = [q, (1-q)/2, (1-q)/2]
        self.outcome_matrix = [-1, 0, 1, 2, 3, 4, 6]

    def write_transitions(self):
        # Writing the transitions from one state to the next states

        for state, indexes in self.state_dict.items():
            if (self.state_dict[state] == 301):
                continue

            next_state = ""
            transition = "transition"
            curr_balls = int(state[:2])
            curr_runs = int(state[2:-1])
            reward = 0

            if (curr_balls != 0):
                if (state[-1] == '0'):
                    # Middle order batsman on strike

                    for action in self.available_actions:
                        # This is the action we take

                        for index, prob in enumerate(self.action_matrix_dict[action]):
                            # This loops through the possible outcomes
                            runs_scored = self.outcome_matrix[index]
                        # This if means wicket has fallen
                            if (index == 0):
                                next_state = "00" + state[2:]
                                reward = 0

                            else:
                                # The batsman scores some run

                                if ((curr_runs - runs_scored) < 10 and (curr_runs - runs_scored) >= 0):
                                    if (curr_balls - 1 < 10):
                                        next_state = "0" + \
                                            str(curr_balls - 1) + "0" + \
                                            str(curr_runs - runs_scored)

                                    else:
                                        next_state = str(
                                            curr_balls - 1) + "0" + str(curr_runs - runs_scored)

                                elif (curr_runs - runs_scored >= 10):
                                    if (curr_balls - 1 < 10):
                                        next_state = "0" + \
                                            str(curr_balls - 1) + \
                                            str(curr_runs - runs_scored)

                                    else:
                                        next_state = str(
                                            curr_balls - 1) + str(curr_runs - runs_scored)

                                else:
                                    # This condition is game won and runs score
                                    if (curr_balls - 1 < 10):
                                        next_state = "0" + \
                                            str(curr_balls - 1) + "00"

                                    else:
                                        next_state = str(
                                            curr_balls - 1) + "00"

                                if (runs_scored % 2 == 0):
                                    # This means that strike is retained and add the same last flag
                                    next_state += state[-1]

                                else:
                                    # Now toggle the last digit
                                    if (state[-1] == "0"):
                                        next_state += "1"
                                    else:
                                        next_state += "0"

                            if ((curr_balls - 1) % 6 == 0):
                                # Over change scenario
                                if (next_state[-1] == "0"):
                                    next_state = next_state[0:4] + "1"
                                else:
                                    next_state = next_state[0:4] + "0"

                            # print("After over change", next_state)
                            if (curr_runs - runs_scored <= 0):
                                reward = 1
                                # print("Reward changed here",
                                #       self.state_dict[state])
                            else:
                                reward = 0

                            transition = "transition " + str(self.state_dict[state]) + " " + \
                                str(self.action_dict[action]) + " " + str(
                                self.state_dict[next_state]) + " " + str(reward) + " " + str(prob)
                            print(transition)

                else:
                    # Tailender is at strike
                    for index, prob in enumerate(self.tailender_actions):
                        runs_scored = self.outcome_matrix[index]
                        if (index == 0):
                            # Tailender is out
                            next_state = "00" + state[2:]
                            reward = 0

                        else:

                            if ((curr_runs - runs_scored) < 10 and (curr_runs - runs_scored) >= 0):
                                if (curr_balls - 1 < 10):
                                    next_state = "0" + \
                                        str(curr_balls - 1) + "0" + \
                                        str(curr_runs - runs_scored)

                                else:
                                    next_state = str(
                                        curr_balls - 1) + "0" + str(curr_runs - runs_scored)

                            elif (curr_runs - runs_scored >= 10):
                                if (curr_balls - 1 < 10):
                                    next_state = "0" + \
                                        str(curr_balls - 1) + \
                                        str(curr_runs - runs_scored)

                                else:
                                    next_state = str(
                                        curr_balls - 1) + str(curr_runs - runs_scored)

                            else:
                                # This condition is game won and runs score
                                if (curr_balls - 1 < 10):
                                    next_state = "0" + \
                                        str(curr_balls - 1) + "00"

                                else:
                                    next_state = str(
                                        curr_balls - 1) + "00"

                            if (runs_scored % 2 == 0):
                                # This means that strike is retained and add the same last flag
                                next_state += state[-1]

                            else:
                                # Now toggle the last digit
                                if (state[-1] == "0"):
                                    next_state += "1"
                                else:
                                    next_state += "0"

                        if ((curr_balls - 1) % 6 == 0):
                            # Over change scenario
                            if (next_state[-1] == "0"):
                                next_state = next_state[:4] + "1"
                            else:
                                next_state = next_state[:4] + "0"

                        if (curr_runs - runs_scored <= 0):
                            reward = 1
                        else:
                            reward = 0

                        transition = "transition " + str(self.state_dict[state]) + " " + \
                            "5" + " " + str(
                            self.state_dict[next_state]) + " " + str(reward) + " " + str(self.tailender_actions[index])

                        print(transition)

    def write_info(self):
        print("numStates " + str(self.num_states))
        print("numActions " + str(self.num_actions))
        print(f"end {self.num_states-2} {self.num_states-1}")

    def write_end(self):
        print("mdptype episodic")
        print("discount 1")


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

    encoder = Encoder(args.states, args.parameters, args.q)
    encoder.write_info()
    encoder.write_transitions()
    encoder.write_end()
