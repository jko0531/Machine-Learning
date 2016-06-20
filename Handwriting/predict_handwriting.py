from __future__ import print_function # this is for python 2.6 <-> 3.x compatibility
import sys
import os, struct
import numpy as np
import matplotlib
from PIL import Image, ImageDraw
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

version = sys.version_info[0]
if version >= 3:
	import tkinter as tk
else:
	import Tkinter as tk

global_matrix = np.zeros([1, 900])

class CanvasEvents:
	def __init__(self, parent=None):
		canvas = tk.Canvas(width=300, height=300, bg='white')
		canvas.pack()
		canvas.bind('<ButtonPress-1>', self.onClick)
		canvas.bind('<ButtonRelease-1>', self.onRelease)
		canvas.bind('<Motion>', self.onMove)
		canvas.bind('<Double-1>', self.onClear)

		# member variables
		self.canvas = canvas
		self.button_status = 'up'
		self.xold, self.yold = None, None

		self.image1 = Image.new('RGB', (300, 300), 'white')
		self.draw = ImageDraw.Draw(self.image1)


	def onClick(self, event):
		self.button_status = 'down'

	def onRelease(self, event):
		self.button_status = 'up'
		self.xold = None
		self.yold = None

	def onMove(self, event):
		if self.button_status == 'down':
			if self.xold is not None and self.yold is not None:
				event.widget.create_line(self.xold, self.yold, event.x, event.y, smooth=True, width=10)
				self.draw.line([self.xold, self.yold, event.x, event.y], 'black', width=10)
			self.xold = event.x
			self.yold = event.y

	def onClear(self, event):
		global global_matrix
		#self.canvas.postscript(file='test.ps', colormode='color')
		size = 30, 30
		filename = 'data/testdraw.png'
		self.image1.thumbnail(size, Image.ANTIALIAS)
		self.image1.save(filename)

		temp = np.array(self.image1)
		final = np.zeros(shape=(30,30))
		for i in range(30):
			for j in range(30):
				if temp[i][j][0] != 255 or temp[i][j][1] != 255 or temp[i][j][2] != 255:
					final[i][j] = 1
				else:
					final[i][j] = 0
		final = final.reshape(1, 30*30)
		#printShape(0, final, ['test'])
		global_matrix = np.concatenate((global_matrix, final), axis=0)
		event.widget.delete('all')
	
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
		raise(ValueError, "'dataset' must be in testing or 'training'")

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
	#X[X > 1] = 1
	return X, y


def printShape(index, X, y):
	'''
	printShape - Given an index, print the pixel image of that handwritten number
	Input - index, X (matrix of handwritings)
	Output - Printed 28 x 28 of the handwritten image

	'''
	counter = 0
	for i in range(30):
		for j in range(30):
			if X[index, counter] == 0:
				print('.', end='  ')
			else:
				print('o', end='  ')
			counter+=1
		print("\n")
	print("Actual result: ", y[index])

def train(correct_number, final, X, y):
	X = np.concatenate((X, final), axis=0)
	y = np.append(y, correct_number)

	np.savetxt('data/self_training_X.csv', X, delimiter=',')
	np.savetxt('data/self_training_y.csv', y, delimiter=',')
	printShape(0, final, [correct_number])


def Main():
	global global_matrix
	#for i in range(100):
	#	print(i+1,"iteration")
	CanvasEvents()
	tk.mainloop()
	temp = np.array(Image.open('data/testdraw.png'))
	final = np.zeros(shape=(30,30))
	for i in range(30):
		for j in range(30):
			if temp[i][j][0] != 255 or temp[i][j][1] != 255 or temp[i][j][2] != 255:
				final[i][j] = 1
			else:
				final[i][j] = 0
	final = final.reshape(1, 30*30)

	global_matrix = np.delete(global_matrix, 0, 0)
	directory = 'data'
	X_test, y_test = parseData('testing', directory)
	X_train, y_train = parseData('training', directory)

	#np.savetxt('self_training.csv', global_matrix, delimiter=',')
	X = np.genfromtxt('data/self_training_X.csv', delimiter=',')
	y = np.genfromtxt('data/self_training_y.csv', delimiter=',')

	#np.savetxt('self_training_y.csv', y, delimiter=',')
	predict_number = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X,y).predict(final)
	print("You drew the number:", int(predict_number[0]))
	if (version > (3,0)):
		response = input("Was I right? (y/n): ")
	else:
		response = raw_input("Was I right? (y/n): ")

	if response == "y" or response == 'yes':
		print("yayyy")
	elif response == 'n' or response == 'no':
		if version >= 3:
			correct_number = input("terribly sorry. What was the correct number?: ")
		else:
			correct_number = raw_input("terribly sorry. What was the correct number?: ")
		train(int(correct_number), final, X, y)
		print("Thanks! We will try to get the correct answer next time :)")


if __name__ == '__main__':
	Main()




