import numpy as np 
import operator
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score


#Import the MNIST data set
mnist_dataset = load_digits()


class KNN:
	"""A class to manage the functionalities of the KNN model"""


	def __init__(self, k, x_test, y_test):
		"""Initialize all the basic attributes of the model"""

		# Import note: To distinguish between the number of folds and the k 
		# for neighbours, a lower case k will be used for neighbours
		self.k = k

		# Fit the data for the model
		self.X_test = x_test
		self.Y_test = y_test

	def euclidean_function(self, point1, point2):
		"""Determine the distance between two input points (white space versus 
		black space) for later comparision against test case"""

		distance = np.sqrt(np.square(point2-point1))
		return distance

	def predict_number(self, X_test, Y_test):
		"""Predict the number fed to the algorithm"""

		#Empty list to store all the prediction
		predictions = []

		# Go through every value of X_test
		for i in range(len(X_test)):





