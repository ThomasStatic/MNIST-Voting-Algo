import numpy 
import operator
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score


class KNN:
	"""A class to manage the functionalities of the KNN model"""


	def __init__(self, x_train, y_train, k = 3):
		"""Initialize all the basic attributes of the model"""

		# Import note: To distinguish between the number of folds and the k 
		# for neighbours, a lower case k will be used for neighbours
		self.k = k

		# Fit the data for the model
		self.X_train = x_train
		self.Y_train = y_train

		# Default the print predictions flag to false
		self.print_predictions = 0
		print("(KNN class) Default setting is to not print the predictions!")


	def euclidean_function(self, point1, point2):
		"""Determine the distance between two input points (white space versus 
		black space) for later comparision against test case"""

		distance = numpy.sqrt(numpy.sum((point2-point1)**2))
		return distance

	def predict_number(self, X_test, Y_test, loops = 0):
		"""Predict the number fed to the algorithm"""

		# Empty list to store all the prediction
		predictions = []

		# If the user didn't specify an amount of digits to predict, loop 
		# through the entire set of X_test
		if loops == 0:
			loops = len(X_test)

		# Go through every value of X_test
		for i in range(loops):

			# For each index of the test set, create an array of the distances
			# between the testing and the training data
			distances = numpy.array([self.euclidean_function(X_test[i], x_train) for 
				x_train in self.X_train])

			# Sort the distances by slicing the list by the value of k
			distances_sorted = distances.argsort()[:self.k]

			# Tuple to store the number of neighbours near each data point tested
			neighbour_count = {}

			# Cycle through every index of the sorted distances list
			for j in distances_sorted:

				# If the training value from the labels is the same as the 
				# value in neighbour count (as in, same amount of neighbours)
				# then increase the value of that index
				if self.Y_train[j] in neighbour_count:
					neighbour_count[self.Y_train[j]] += 1

				# IMPORTANT: This is setting the default value in the tuple to 1
				else:
					neighbour_count[self.Y_train[j]] = 1

				# Grab the total tally of the neighbour count at this index so 
				# we can use/display the prediction later
				sorted_neighbour_count = sorted(neighbour_count.items(), key = 
					operator.itemgetter(1), reverse = True)
				predictions.append(sorted_neighbour_count[0][0])

				if self.print_predictions == 1:
					print(f"\nKNN Prediction: {sorted_neighbour_count[0][0]}")
					print(f"Actual Number: {Y_test[i]}")

		return predictions

	def _change_print_flag(self):
		"""Change the flag for whether or not predictions are printed"""

		# If originally the program was set not to print results, now print
		if self.print_predictions == 0:
			self.print_predictions = 1
			print("Prediction results will now be printed!")
			return 1

		# If originally the program was set to print results, no longer print
		if self.print_predictions == 1:
			self.print_predictions = 0
			print("Prediction results will no longer be printed!")
			return 0


#Import the MNIST data set
mnist_dataset = load_digits()

X = mnist_dataset.data
y = mnist_dataset.target

# NOTE: You can randomize this split with a seed using random_state = INT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

setup_model = KNN(X_train, y_train, k = 3)




