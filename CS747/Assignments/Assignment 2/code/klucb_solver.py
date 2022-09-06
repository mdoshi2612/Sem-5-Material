import numpy as np
import math

def KL(p, q):
	if (p == 1):
		return math.log(1/q)
	
	elif(p == 0):
		return math.log((1)/(1-q))
	
	return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))

def solver(mean, time, count, c):
	THRESHOLD = 0.001
	STEP = 0.01
	q = mean + STEP
	rhs = (math.log(time) + c*math.log(math.log(time)))/count
	while(KL(mean, q) - rhs < 0 and q < 0.99):
		print(KL(mean, q) - rhs)
		q += STEP   
	return q

def solve_q(rhs, p_a):
	if p_a == 1:
		return 1 
	q = np.arange(p_a, 1, 0.01)
	lhs = []
	for el in q:
		lhs.append(KL(p_a, el))
	print(lhs)
	lhs_array = np.array(lhs)
	lhs_rhs = lhs_array - rhs
	lhs_rhs[lhs_rhs <= 0] = np.inf
	print(lhs_rhs)	
	min_index = lhs_rhs.argmin()
	return q[min_index]
