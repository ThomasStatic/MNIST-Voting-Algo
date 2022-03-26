# importing the libraries needed to run the algorithm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Importing Sickit Learn Modules
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import validation_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Read in testing and training csv files
train_data = pd.read_csv("training.csv")
test_data = pd.read_csv("testing.csv")

# Print out data headers and shapes
print(train_data.shape)
print(test_data.shape)
print(train_data.head)
print(test_data.head)


# print data dimensions
print("Dimensions: ",test_data.shape, "\n")

# print data types
print(test_data.info())

# Print columns
print(train_data.columns)
print(test_data.columns)

# oder data into list from 0 to 9
order = list(np.sort(train_data['label'].unique()))
print(order)

# 4 testing and training printed to plot
four = train_data.iloc[3, 1:]
four.shape
four = four.values.reshape(28,28)
plt.imshow(four, cmap='gray')
plt.title("Digit 4")

# 7 testing and training printed to plot
seven = train_data.iloc[6, 1:]
seven.shape
seven = seven.values.reshape(28, 28)
plt.imshow(seven, cmap='gray')
plt.title("Digit 7")

# normalizing data
round(train_data.drop('label', axis=1).mean(), 2)

# set label to x and y values
y = train_data['label']

X = train_data.drop(columns = 'label')

# Printing the size of data
print(train_data.shape)

# Using rbf kernel to create a non-linear model   
non_linear_model = SVC(kernel='rbf')

# fit data to model
non_linear_model.fit(X_train, y_train)

# predict based on y vanlue the value
y_pred = non_linear_model.predict(X_test)