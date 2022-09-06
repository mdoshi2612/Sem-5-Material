import numpy as np
import math


def KL(p, q):
    if (p == 1):
        return math.log(1/q)

    elif (p == 0):
        return math.log((1)/(1-q))

    return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))


def rhs(time, count, c=3):
    return (math.log(time) + c*math.log(math.log(time)))/count


def solver(p, rhs, STEP=0.01):
    q = 1-1e-5
    while (q >= p and (KL(p, q) - rhs > 0)):
        q -= STEP
    return q


print(solver((0.5), rhs(1000, 900, 3)))
