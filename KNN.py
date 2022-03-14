# Import Libraries
import pandas
import csv
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot
import matplotlib.pyplot as plt
import operator
import seaborn
from sklearn import svm, datasets
import matplotlib.patches as mpatches
#from KNN_Class import KNN

# import data through tensorflow

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# determine if pixel should be black or white
def blackOrWhite(X):
    # function takes in original color of surrounding pixels of a number and shifts them to white/black rather than in between
    middle = 125
    k = np.empty_like(X)
    k[X < middle] = 0
    k[X >= middle] = 1
    k = k.reshape(*k.shape, 1)
    return k

# set variables for testing and training label arrays
test_labels = test_y
train_labels = train_y

# Split testing labels into smaller arrays to improve performance and to fit to csv file
split1 = np.split(test_labels, 4)

# # open testing.csv to write mode
f = open('testing.csv', 'w')

# create loop to fill testing.csv with label values
for i in range(4):
    writer = csv.writer(f)
    writer.writerow(split1[i])
    i = i + 1
    print("\n")

# close training.csv
f.close()

# open training.csv to write mode
f = open('training.csv', 'w')

# Split training labels into smaller arrays to improve performance and to fit to csv file
split2 = np.split(train_labels,10)

# create loop to fill training.csv with label values
for j in range(10):
    writer = csv.writer(f)
    writer.writerow(split2[j])
    j = j + 1
    print("\n")

# close training.csv
f.close()


# train and tests sets are put through black/white function
train_X = blackOrWhite(train_X)
test_X = blackOrWhite(test_X)

# plot images imported through tensorflow (example set displaying only 9 images)
for i in range (9):
    pyplot.subplot(330+1+i)
    #print(f"Example number: {i+1} ")
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    #plt.show()


def euclidean_distance(point1, point2):
        """Determine the distance between reference point and number's point"""

        distance = 0

        # Square the sum of all dimension equal values
        #for x in range(dimension):
        distance += np.square(point1 - point2)

        # Return the root value of this sum
        return np.sqrt(distance)


class KNN:
    """A class for handling the KNN functionalities of the program"""


    def __init__(self, k = 3):
        """ Initialize all attributes of KNN function"""
        # NOTE: k value defaulted to k = 3
        self.k = k

    def fit(self, x_train, y_train):
        """Add the training data as an attribute of the class"""
        self.X_train = x_train
        self.Y_train = y_train


    

    def predict_number(self, X_test):
        """Predict the value of the currently tested number"""

        # Create an empty list to store the predictions
        predictions = []

        # Ensures we are only using 1 axis
      #  dimension = X_test.shape[1]

        # Find an x for each individual value of x we find
        for i in range(len(X_test)):
            distance = np.array([euclidean_distance(X_test[i], x_t) for x_t in 
                self.X_train])
            distance.tolist()
            distance.tolist()


            # Sort the distances out to a useable format
            dist_sorted = distance.argsort()[:self.k]
             



            # Empty tuple for storing neighbour counts
            neighbour_count = {}

            for index in dist_sorted:
                if self.Y_train[index] in neighbour_count:
                    neighbour_count[self.Y_train[index]] += 1
                else:
                    neighbour_count[self.Y_train[index]] = 1

            # Sort the above results for number of neighbours
            sorted_neighbour_count = sorted(neighbour_count.items(), 
                key = operator.itemgetter(1), reverse = True)
            predictions.append(sorted_neighbour_count[0][0])
        
        return predictions


knn = KNN(k = 3) #NOTE: We should add a function that determines the optimal k value rather than relying on the default k = 3

knn.fit(train_X, train_y)
pred = knn.predict_number(test_X)
print(pred)





