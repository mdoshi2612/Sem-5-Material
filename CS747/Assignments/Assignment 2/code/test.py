import numpy as np
import math
from unittest.util import _count_diff_hashable
from bernoulli_bandit import BernoulliBandit


def first_pass_UCB():
    bandit = BernoulliBandit(
        probs=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    rewards = [bandit.pull(i) for i in range(bandit.num_arms())]
    # for i in range(bandit.num_arms()):
    #     rewards.append(bandit.pull(i))
    print(rewards)

# first_pass_UCB()


values = np.zeros(5)
counts = np.ones(5)
time = 1
num_arms = 5


def partition(batch_size):
    first_elem = int(0.4*batch_size)
    second_elem = int(0.25*batch_size)
    third_elem = int(0.15*batch_size)
    fourth_elem = int(0.1*batch_size)
    fifth_elem = batch_size - \
        (first_elem + second_elem + third_elem + fourth_elem)
    return np.array([first_elem, second_elem, third_elem, fourth_elem, fifth_elem], dtype=np.int64)


def ucb():
    ucb = [values[i] + math.sqrt((2*math.log(time)/counts[i]))
           for i in range(num_arms)]


epsilon = 1e-3
# print(ucb)

num_arms = 10
batch_size = 10
success = np.zeros(num_arms)
failure = np.zeros(num_arms)
samples = np.zeros(num_arms*batch_size)

# print(solver(0.6, 80, 4, 3))
# for arm_index in range(num_arms):
#     samples[arm_index] = []

np.random.seed(0)

samples = np.array([np.random.beta(
    success[i]+1, failure[i]+1, size=batch_size) for i in range(num_arms)])
print(samples)
print("flatten array")
samples = np.squeeze(samples.ravel())
print(samples)
temp = (np.argsort(samples)[-batch_size:]/10).astype(int)
print(temp)

unique, counts = np.unique(temp, return_counts=True)
print(unique)
print(counts)


# time = 80
# rhs = (math.log(time) + 3*math.log(math.log(time)))/8


# print(solve_q(rhs, 0.5))
