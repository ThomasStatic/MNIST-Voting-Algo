import numpy as np


class KNN:
	"""A class for handling the KNN functionalities of the program"""


	def __init__(self, k = 3):
		""" Initialize all attributes of KNN function"""
		# NOTE: k value defaulted to k = 3
		self.k = k

	def fit(self, x_train, y_train):
		"""Add the training data as an attribute of the class"""
		self.x_train = x_train
		self.y_train = y_train


	def euclidean_distance(point1, point2, dimenion):
		"""Determine the distance between reference point and number's point"""

		distance = 0

		for x in range(dimension):
			distance += np.square(point1[x] - point2[x])

		return np.sqrt(distance)



