import numpy as np
import math
from sklearn.neural_network import MLPRegressor

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

# Mathematical Function Y = A0 + A1 X + A2 X X
def estimateFunction(x, clf):
	X = np.array([x])
	X = X.reshape(1,-1)
	return clf.predict(X)


# Main Method
def Li_Square_NN(n_paths, p_length, S_zero, K, r, volatility):
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

	clf = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(5, 2))

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
		X = np.asarray(X, dtype=np.float64)
		Y = np.asarray(Y, dtype=np.float64)
		X = X.reshape(-1, 1)
		clf.fit(X, Y)

		for k in range(len(indicies)):
			if estimateFunction(paths[indicies[k]][i], clf) < payoff[indicies[k]][i]:
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

	Li_Square_NN(n_paths, p_length, S_zero, K, r, volatility)