import numpy as np
import pandas as pd

class NeuralNetwork:

	def __init__(self, X, y):
		m, n = X.shape
		self.X = X
		self.y = y
		self.hidden_layer_size = 25
		self.input_layer_size = n
		self.output_layer_size = 10

	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(np.negative(z)))

	def sigmoidGradient(self, z):
		return self.sigmoid(z) * (1 - self.sigmoid(z))

	def reshapeParams(self, flattened_array):
		theta1 = flattened_array[:(self.input_layer_size+1)*self.hidden_layer_size] \
				.reshape((self.hidden_layer_size,self.input_layer_size+1))
		theta2 = flattened_array[(self.input_layer_size+1)*self.hidden_layer_size:] \
				.reshape((self.output_layer_size,self.hidden_layer_size+1))
		
		return [ theta1, theta2 ]

	def genRandThetas(self):
		epsilon_init = 0.12
		theta1_shape = (self.hidden_layer_size, self.input_layer_size+1)
		theta2_shape = (self.output_layer_size, self.hidden_layer_size+1)
		rand_thetas = [ np.random.rand( *theta1_shape ) * 2 * epsilon_init - epsilon_init, \
						np.random.rand( *theta2_shape ) * 2 * epsilon_init - epsilon_init]
		return rand_thetas


	def costFunction(self, X, y, theta1, theta2, lambd):

		## do feed foward propogation to get the h_x vector ##

		# get m & n variables
		m, n = X.shape

		# first must set a = X. (5000 x 401)
		a = X
		#a = np.c_[np.ones(m), X]

		# theta1 is (25 x 401), theta2 is (10 x 26)
		# then, z2 is theta1 * a1. z2 has dimensions (5000 x 25)
		z2 = a.dot(theta1.T)
		# a2 is sigmoid(z2), add row of 1s to a2 (5000 x 26)
		a2 = np.c_[np.ones(m), self.sigmoid(z2)]
		# z3 is theta2 * a2. z3 has dimensions (5000 x 10)
		z3 = a2.dot(theta2.T)
		# a3 is sigmoid(z3). dimensions (5000 x 10) h == a3
		h = self.sigmoid(z3)

		## cost function ##

		# turn y into matrix of logical arrays
		y_matrix = pd.get_dummies(y.ravel()).as_matrix()

		# cost, the theta1/theta2 slice is to ignore first column for regularization
		cost = np.sum((-(y_matrix) * np.log(h)) - (1 - y_matrix) * np.log(1 - h))/m + \
				(lambd/(2.0*m)) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))
		## gradient ##

		error3 = h - y_matrix     # shape is (5000 x 10)

		error2 = theta2.T.dot(error3.T)
		error2 = error2[1:, :] * self.sigmoidGradient(z2.T)

		# error2 shape is (25 x 5000)

		delta1 = error2.dot(a)
		delta2 = error3.T.dot(a2)
		theta1_ = np.c_[np.ones((theta1.shape[0],1)),theta1[:,1:]]
		theta2_ = np.c_[np.ones((theta2.shape[0],1)),theta2[:,1:]]
		
		theta1_grad = delta1/m + (theta1_*lambd)/m
		theta2_grad = delta2/m + (theta2_*lambd)/m


		return cost, theta1_grad, theta2_grad


	def gradientDescent(self, X, y, theta1, theta2, iterations):
		for i in range(iterations):
			cost, grad1, grad2 = self.costFunction(X, y, theta1, theta2, .01)
			theta1 = theta1 - grad1
			theta2 = theta2 - grad2

		return cost, theta1, theta2



	def predict(self, theta_1, theta_2, features):
		z2 = theta_1.dot(features.T)
		a2 = np.c_[np.ones((features.shape[0],1)), self.sigmoid(z2).T]
		
		z3 = a2.dot(theta_2.T)
		a3 = self.sigmoid(z3)

		return(np.argmax(a3, axis=1)+1)

	def fit(self, X, y, final):
		m, n = X.shape
		M, N = final.shape
		X = np.c_[np.ones(m), X]
		theta1, theta2 = self.genRandThetas()
		cost, theta1, theta2 = self.gradientDescent(X, y, theta1, theta2, 100)
		final = np.c_[np.ones(M), final]
		return self.predict(theta1, theta2, final)
