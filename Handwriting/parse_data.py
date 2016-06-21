'''
Jason Ko
This program parses the MNIST database into easily readable matrices

'''
from __future__ import print_function # this is for python 2.6 <-> 3.x compatibility
import os, struct
import numpy as np
import math
from scipy import optimize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def parseData(dataset='testing', path='.'):
	'''
	parseData - Parses a file into matrices
	Input - the name of file to be parsed
	Output - The data in matrix representation

	'''

	if dataset == 'training':
		image_file = os.path.join(path, 'train-images-idx3-ubyte')
		label_file = os.path.join(path, 'train-labels-idx1-ubyte')
	elif dataset == 'testing':
		image_file = os.path.join(path, 't10k-images-idx3-ubyte')
		label_file = os.path.join(path, 't10k-labels-idx1-ubyte')
	else:
		raise ValueError, "'dataset' must be in testing or 'training'"

	# get the matrix for image data
	f_img = open(image_file, 'rb')
	magic_nr, size = struct.unpack(">II", f_img.read(8))  # parse the magic number, & size of dataset
	dim_x, dim_y = struct.unpack(">II", f_img.read(8))  # get the dimensions of each handwritten num
	X = np.fromfile(f_img, dtype=np.dtype('B'))
	X = X.reshape(size, dim_x * dim_y)


	# get the matrix for label data
	f_lbl = open(label_file, 'rb')
	magic_nr, size = struct.unpack(">II", f_lbl.read(8)) # only magic # and size of dataset
	y = np.fromfile(f_lbl, dtype=np.dtype('B'))

	return X, y

def printShape(index, X, y):
	'''
	printShape - Given an index, print the pixel image of that handwritten number
	Input - index, X (matrix of handwritings)
	Output - Printed 28 x 28 of the handwritten image

	'''
	counter = 0
	for i in range(28):
		for j in range(28):
			if X[index, counter] == 0:
				print('.', end='  ')
			else:
				print('o', end='  ')
			counter+=1
		print("\n")
	print("Actual result: ", y[index])

def sigmoid(z):
	## note... the sigmoid function is not efficient for gradient descent 
	## with iterations...takes too long
	return 1 / (1 + np.exp(-z))

def map_feature(x1, x2):
    '''
    Maps the two input features to quadratic features.
    Returns a new feature array with more features, comprising of
    X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, etc...
    Inputs X1, X2 must be the same size
    '''
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)
    degree = 6
    out = np.ones(shape=(x1[:, 0].size, 1))

    m, n = out.shape

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)

    return out

def cost_function_reg(theta, X, y, l):
    '''Compute the cost and partial derivatives as grads
    '''
    m, n = X.shape
    h = sigmoid(X.dot(theta))

    thetaR = theta[1:, 0]

    J = (1.0 / m) * ((-y.T.dot(np.log(h))) - ((1 - y.T).dot(np.log(1.0 - h)))) \
            + (l / (2.0 * m)) * (thetaR.T.dot(thetaR))

    delta = h - y
    sumdelta = delta.T.dot(X[:, 1])
    grad1 = (1.0 / m) * sumdelta

    XR = X[:, 1:X.shape[1]]
    sumdelta = delta.T.dot(XR)

    grad = (1.0 / m) * (sumdelta + l * thetaR)

    out = np.zeros(shape=(grad.shape[0], grad.shape[1] + 1))

    out[:, 0] = grad1
    out[:, 1:] = grad

    return J.flatten(), out.T.flatten()


def costFunction(theta, X, y, lambd):
	'''
	gradientDescent - Computes gradient descent
	Input - Bunch of stuff
	Output - Updated theta

	'''
	
	m = len(y)
	h = sigmoid(X.dot(theta))
	grad = np.zeros(len(theta))
	#error = -(y * np.array(np.log(h))) - (1-y) * np.array(np.log(1-h))
	error = -(y * np.array(np.log(h))) - (1-y) * np.array(np.log(1-h))
	temp = theta
	temp[0] = 0
	cost = (1.0/m) * np.sum(error) + (lambd/(2.0*m) * np.sum(np.power(temp,2)))
	grad = (X.T.dot((h-y)))/m
	grad = grad + ((lambd/m) * temp)

	return cost, grad




def oneVsAll(X, y, num_labels, lambd):

	# variables m and n for matrix X (m x n array)
	# m is the # of samples, n is the # of features
	m, n = X.shape

	# iterations
	#iterations = 100

	# add ones to the X data matrix
	X = np.c_[np.ones(m), X]
	
	# initialize fitting parameter theta
	#all_theta = np.zeros((num_labels,n+1))


	# initialize theta for loop
	initial_theta = np.zeros(n+1)

	# get correct shape for y, and theta
	initial_theta.shape = (n+1, 1)
	y.shape = (m, 1)

	#print(initial_theta.shape)
	cost, initial_theta = costFunction(initial_theta, X, y, lambd)
	print(initial_theta)
	#cost, initial_theta = costFunction(initial_theta, X, y, lambd)


	print(cost)




def test1():
	theta = np.array([-2,-1,1,2])
	y = np.array([1,0,1,0,1]) >= .5
	X = np.matrix([[1, .1, .6, 1.1], [1, .2, .7, 1.2], [1, .3, .8, 1.3], [1, .4, .9, 1.4], [1, .5, 1, 1.5]])
	(m, n) = X.shape
	lambd = 3.0
	theta.shape = (n, 1)
	y.shape = (m, 1)
	cost, theta = costFunction(theta, X, y, lambd)
	cost, theta = costFunction(theta, X, y, lambd)
	cost, theta = costFunction(theta, X, y, lambd)
	print(cost)
	print(theta.shape)
	print(theta)



def Main():
	directory = 'data'
	X_train, y_train = parseData('training', directory)
	X_test, y_test = parseData('testing', directory)
	num_labels = 10
	lamd = 0.1

	a = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X_test,y_test)#.predict(X_train)
	#print(a)
	print(a.score(X_train, y_train))
	#printShape(45, X_train, a)
	#print(X.shape)
	#print(y.shape)
	#printShape(59998, X, y)
	#all_theta = oneVsAll(X, y, num_labels, lamd)


	#test1()




if __name__ == '__main__':
	Main()

