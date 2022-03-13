# Import Libraries
import csv
import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot
import matplotlib.pyplot as plt

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
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
    plt.show()
