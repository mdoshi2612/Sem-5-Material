import numpy as np
import math
import time


def KL(p, q):
    if (p == 1):
        return math.log(1/q)

    elif (p == 0):
        return math.log((1)/(1-q))

    return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))


def rhs(time, c=3):
    return math.log(time)+c*math.log(math.log(time))


def solver(mean, time, count, c=3):
    bound = rhs(time, c)/count
    q = mean
    step = (1-mean)/2
    while step > 1e-6:
        if (KL(mean, q + step) <= bound):
            q += step
        step /= 2
    return q


start = time.time()
q = solver(0.5, 100, 10)
end = time.time()
print(q)
print(end - start)
