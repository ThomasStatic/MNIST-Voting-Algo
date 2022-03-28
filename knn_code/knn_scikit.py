#%matplotlib inline    NOTE: Uncomment this if running in Jupyter Notebook
import pandas as pandas
import numpy as numpy
import matplotlib.patches as mpatches
from sklearn import svm, datasets
import matplotlib.pyplot as mplplt

#This is a data set with a bunch of identifying features of flower species
iris = datasets.load_iris()

X = iris.data[:, :2] #Take the first two features
y = iris.target # target = dependent variable

colors = ['red', 'green', 'blue']
#Creates a scatter plot for values 0 through 1 which display a coressponding colour for each class
for color, i, target in zip(colors, [0, 1, 2], iris.target_names):
	mplplt.scatter(X[y==i, 0], X[y==i, 1], color = color, label = target)
 
# Create names for the x and y axes
mplplt.xlabel('Sepal length')
mplplt.ylabel('Sepal width')
mplplt.legend(loc = 'best', shadow = False, scatterpoints = 1)

# Give the graph a name
mplplt.title('Scatter plot of Sepal width against Sepal length')
mplplt.show()

# BELOW HERE IS THE IMPLEMETNATION OF KNN ON THIS DATA

from sklearn.neighbors import KNeighborsClassifier

k = 13
# Create an instant of the class
knn = KNeighborsClassifier(n_neighbors = k)

# Fit the data for the model
knn.fit(X, y)

# Find the minimum and maximum values for Sepal length
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

# Find the min and max values for Sepal height
y_min, y_max = X[:, 1].min() -1, X[:, 1].max() + 1

# Set the pattern step size for the data
h = (x_max / x_min) /100

# Make predictions for each of the points in xx, yy
# meshgrid() makes a (rectangular) grid out of one or two dimensional arrays
# arrange() orders the rows of a data frame
xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
						numpy.arange(y_min, y_max, h))


# numpy.c concatenates two rows of a matric together
# ravel() flatens out an array and makes it continous
Z = knn.predict(numpy.c_[xx.ravel(), yy.ravel()])

# Now we draw the result (to make this program user friendly)
Z = Z.reshape(xx.shape)
mplplt.contourf(xx, yy, Z, cmap = mplplt.cm.Accent, alpha = 0.8)

# Plot the training points:
colors = ['red', 'green', 'blue']
# zip() takes in interators (think anything from a tuple to files, etc) and returns an iterator
for color, i, target in zip(colors, [0,1,2], iris.target_names):
	mplplt.scatter(X[y==i, 0], X[y==i, 1], color = color, label = target)

mplplt.xlabel('Sepal length')
mplplt.ylabel('Sepal width')
mplplt.title(f'KNN (k = {k})')
mplplt.legend(loc = 'best', shadow = False, scatterpoints = 1)

mplplt.show()

predictions = knn.predict(X)

# Print the classification based on the prediction
print(numpy.unique(predictions, return_counts = True))

# BELOW HERE IS MODEL TRAINING AND VALIDATION CODE

from sklearn.model_selection import cross_val_score

# Holds the cross validated scores
cv_scores = []

# Use all the flower features
X = iris.data[:, :4]
y = iris.target

# Number of folds (think of a folds as layers of a lasagna)
# We want the program to cross validate using a rotation of different layers
folds = 10

# Create an odd list of K for KNN
# Ensuring K is an odd number prevents ties
ks = list(range(1, int(len(X) * ((folds-1)/folds))))

# Remove all the multiples of 3
ks = [k for k in ks if k % 3 != 0]

# Perform k-fold cross validation
for k in ks:
	knn = KNeighborsClassifier(n_neighbors=k)

	# Perform cross validation and return the average accuracy
	#cross_val_score() performs cross validation automatically and returns metrics
	scores = cross_val_score(knn, X, y, cv = folds, scoring = 'accuracy')
	mean = scores.mean()
	cv_scores.append(mean)
	print(k, mean)

# BELOW IS THE CODE FOR DETERMINING THE OPTIMAL K VALUE

# MSE is abbreviated form of missclassification error
MSE = [1-x for x in cv_scores]

# The optimal k will be where ever the MSE is the lowest value
optimal_k = ks[MSE.index(min(MSE))]
print(f"The optimal number of neighbours is {optimal_k}")

# Plot the misclassification error vs k for user friendliness
mplplt.plot(ks, MSE)
mplplt.xlabel("Number of neighbours K")
mplplt.ylabel("Misclassification Error")
mplplt.show()