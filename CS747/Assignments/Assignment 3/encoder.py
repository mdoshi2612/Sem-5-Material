import argparse
import sys
import os
import numpy as np


class Encoder():
    def __init__(self, statefile_path, parameters_path, q):
        self.action_dict = {'0': 0, '1': 1, '2': 2, '4': 3, '6': 4}
        self.outcome_dict = {'-1': 0, '0': 1,
                             '1': 2, '2': 3, '3': 4, '4': 5, '6': 6}

        self.q = q
        self.num_actions = 6

        self.state_path = statefile_path
        self.action_prob_matrix = np.zeros((self.num_actions, 7))
        self.tail = False

        with open(self.state_path, 'r') as statefile:
            line = statefile.readline()
            balls, runs = int(line[:2]), int(line[2:])
            self.num_states = balls*runs*2 + 2
            self.hash_param = runs
            self.half_states = balls*runs
            self.win_state = self.num_states - 1
            self.loss_state = self.num_states - 2
        with open(parameters_path, 'r') as file:
            lines = file.readlines()[1:]

            for line in lines:
                action = line.split()[0]
                self.action_prob_matrix[self.action_dict[str(line[0])]] = (line.split()[
                    1:])

    def write_info(self):
        print("numStates " + str(self.num_states))
        print("numActions " + str(self.num_actions))
        print(f"end {self.num_states-2} {self.num_states-1}")

    def get_action_key(self, action):
        # This function provides the index in the array and return the action key
        return (list(self.action_dict.keys())
                [list(self.action_dict.values()).index(action)])

    def get_outcome_key(self, outcome):
        # This function provides the index in the array and return the outcome key
        return (list(self.outcome_dict.keys())
                [list(self.outcome_dict.values()).index(outcome)])

    def write_transitions(self):

        # Assuming middle order batsman is on strike
        with open(self.state_path, 'r') as statefile:
            lines = statefile.readlines()
            for line in lines:
                self.tail = False
                # print("Line " + line)
                current_balls = int(line[:2])
                current_runs = int(line[2:])
                # print(
                # f"Current balls and runs are {current_balls} and {current_runs}")
                if (current_balls >= 1 and current_runs >= 1):
                    if self.tail == False:
                        curr_state = self.hash_param * \
                            (current_balls-1) + current_runs-1

                    else:
                        curr_state = self.hash_param * \
                            (current_balls-1) + current_runs-1 + self.half_states

                    if self.tail == True:
                        # This means tailender is at strike position
                        # If tailender is out
                        print(
                            f"transition {curr_state} 5 {self.loss_state} {self.q} 0")

                        # If tailender takes no run
                        next_state = self.hash_param * \
                            (current_balls - 2) + \
                            current_runs - 1 + self.half_states
                        self.tail = True
                        if ((current_balls - 1) % 6 == 0):
                            self.tail ^= True
                            next_state -= self.half_states
                        print(
                            f"transition {curr_state} 5 {next_state} {(1 - self.q)/2} 0")

                        # If tailender takes 1 run
                        if (current_runs == 1):
                            # If the game is done when this one run is taken
                            print(
                                f"transition {curr_state} 5 {self.win_state} {(1 - self.q)/2} 1")
                            self.tail ^= True

                        else:

                            # Game does not finish
                            next_state = self.hash_param * \
                                (current_balls - 2) + current_runs - 2
                            if ((current_balls - 1) % 6 == 0):
                                self.tail ^= True
                                next_state += self.half_states
                            print(
                                f"transition {curr_state} 5 {next_state} {(1 - self.q)/2} 0")

                    else:
                        # This means that one of the middle order batsman is at strike position
                        for action_index, action in enumerate(self.action_prob_matrix[:]):
                            # Now the action is defined

                            for outcome_index, prob in enumerate(action):
                                # Now the different outcomes can be defined here
                                runs_scored = int(
                                    self.get_outcome_key(outcome_index))
                                if (runs_scored == -1):
                                    # If the batsman is out

                                    print(
                                        f"transition {curr_state} {action_index} {self.loss_state} {prob} 0")

                                elif (current_runs - runs_scored <= 0):
                                    # If the game is over
                                    print(
                                        f"transition {curr_state} {action_index} {self.win_state} {prob} 1")

                                elif (current_balls - 1 == 0):
                                    print(
                                        f"transition {curr_state} {action_index} {self.loss_state} {prob} 0")

                                else:
                                    # The game isn't over
                                    if (runs_scored % 2 == 0):
                                        # Batsman Retains Strike
                                        next_state = self.hash_param * \
                                            (current_balls - 2) + \
                                            current_runs - runs_scored - 1
                                        if ((current_balls - 1) % 6 == 0):
                                            self.tail ^= True
                                            next_state += self.half_states
                                        print(
                                            f"transition {curr_state} {action_index} {next_state} {prob} 0")
                                    else:
                                        # Batsman loses strike
                                        self.tail = True
                                        next_state = self.hash_param * \
                                            (current_balls - 2) + \
                                            current_runs - runs_scored - 1 + self.half_states
                                        if ((current_balls-1) % 6 == 0):
                                            self.tail ^= True
                                            next_state -= self.half_states
                                        print(
                                            f"transition {curr_state} {action_index} {next_state} {prob} 0")

                else:
                    pass

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
