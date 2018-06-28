import numpy as np
import math

# Function to generate Paths
def generate_paths(paths, n_paths, p_length, S_zero, r, volatility):
	for j in range(n_paths):
		paths[j][0] = S_zero

	mu = r
	sigma = volatility

	for i in range(n_paths):
		for j in range(1, p_length):
			W = np.random.normal(0, 1, 1)
			paths[i][j] = paths[0][0] * math.exp((mu - ((sigma * sigma) / 2)) * j + sigma * W)

# Function to calculcate A0, A1, A2, ... using Matrix Multiplication
def regression(Y, X, reg_length):
	A = np.zeros((1, reg_length), dtype=float)
	B = np.zeros((reg_length, reg_length), dtype=float)
	C = np.zeros((reg_length, 1), dtype=float)
	l_X = len(X)

	for i in range(reg_length):
		for j in range(reg_length):
			tempSum = 0
			for k in range(l_X):
				tempSum += X[k]**(i + j)
			B[i][j] = tempSum
	
	for j in range(reg_length):
		tempSum = 0
		for k in range(l_X):
			tempSum += Y[k] * (X[k]**j)
		C[j] = tempSum

	A = np.matmul(np.linalg.inv(B), C)
	return A

# Mathematical Function Y = A0 + A1 X + A2 X X
def estimateFunction(x, A):
	tempSum = 0
	for i in range(len(A)):
		tempSum += A[i] * (x ** i)
	return tempSum


# Main Method
def Li_Square(n_paths, p_length, S_zero, K, r, volatility):
	t = 1


	paths = np.zeros((n_paths, p_length), dtype=float)

	generate_paths(paths, n_paths, p_length, S_zero, r, volatility)

	print "========================================================"
	print "Generated Paths:"
	print paths
	print "========================================================"
	print "\n\n"

	payoff = np.zeros((n_paths, p_length), dtype=float)
	cash_flow = np.zeros((n_paths, p_length), dtype=float)
	stopping_rule = np.zeros((n_paths, p_length), dtype=int)

	for j in range(p_length):
		i = p_length - j - 1
		for k in range(n_paths):
			payoff[k][i] = max(0, K - paths[k][i])

	for j in range(n_paths):
		cash_flow[j][p_length - 1] = max(0, K - paths[j][p_length - 1])

	for j in range(p_length - 2):
		i = p_length - 2 - j
		indicies = []
		X = []
		Y = []
		for k in range(n_paths):
			if (payoff[k][i] > 0):
				X.append(paths[k][i])
				Y.append(cash_flow[k][i + 1] * math.exp(- r * t))
				indicies.append(k)

		A = regression(Y, X, 3)

		for k in range(len(indicies)):
			if estimateFunction(paths[indicies[k]][i], A) < payoff[indicies[k]][i]:
				cash_flow[indicies[k]][i] = payoff[indicies[k]][i]
				for y in range(i + 1, p_length):
					cash_flow[indicies[k]][y] = 0
			else:
				cash_flow[indicies[k]][i] = 0

	tempSum = 0
	for j in range(n_paths):
		for i in range(p_length):
			if (cash_flow[j][i] > 0):
				stopping_rule[j][i] = 1
				tempSum += cash_flow[j][i] * math.exp(- r * i)
	
	option_price = tempSum/n_paths

	print "========================================================"
	print "Final Cash Flow:"
	print cash_flow
	print "========================================================"
	print "\n\n"

	print "========================================================"
	print "Stopping Rule:"
	print stopping_rule
	print "========================================================"
	print "\n\n"

	print "Option Price:", option_price



# Example Function Usage
if __name__ == "__main__":
	n_paths = 8
	p_length = 4

	S_zero = 10
	K = 12
	r = 0.1
	volatility = 0.3

	Li_Square(n_paths, p_length, S_zero, K, r, volatility)