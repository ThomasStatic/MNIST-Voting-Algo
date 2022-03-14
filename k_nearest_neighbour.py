# NOTE: THIS CODE IS A TUTORIAL ALTERED FROM THE BOOK PYTHON MACHINE LEARNING
# THIS IS NOT ORIGINAL WORK, THIS IS AN ALTERED TUTORIAL


import pandas
import numpy
import operator
import seaborn 
import matplotlib.pyplot

#Using the pandas library to read the csv file
#CSV file is 'comma seperated values' file
#This text file is written to split up data into two groups, group A and B
data = pandas.read_csv("knn.csv")


#Here's a breakdown of what seaborn.lmplot is doing:
#Plot on the x and y graph
#The data we'll use is the data variable we set (to the csv file) above
#Hue how we're classifying the data points. If you look in the file, these points are split between A and B
#Palette determines the colours used in the graph
#fit_reg "fits" the data based off a mean
#Because we denote "s" in scatter_kws, this means the greater the value given, the greater the plot/node (in this case 70)
seaborn.lmplot('x', 'y', data=data, hue = 'Class (A or B)', 
             palette='Set1',
             fit_reg=False, scatter_kws={"s": 70})



#This plots the graph made in line 24
matplotlib.pyplot.show()

#To calculate the distance between two points, we use the distance formula
#We will do this a lot, so it's worth making a function
def euclidean_distance(pt1, pt2, dimension):
	distance = 0

	#For x in the dimension (2D vs 3D)
	for x in range(dimension):

		#Distance equals the square of position 1 - position 2
		
		distance += numpy.square(pt1[x] - pt2[x])
		#Return the squareroot of this result
	return numpy.sqrt(distance)

#This is our function to actually preform the k nearest neighbour algorithm
def knn(training_points, test_point, k):

	#We initialize distances as an empty tuple
	distances = {}

	#We only want to deal with one axis for now, but this alters the amount of axes we use
	dimension = test_point.shape[1]

	#Find an x for each value we train knn with
	for x in range(len(training_points)):
		#Find the total distance by feed the values of our file into the euclidean_distance function
		#iloc[] finds a specific value for a number from a specific row or column
		dist = euclidean_distance(test_point, training_points.iloc[x], dimension)

		#This records the result 
		distances[x] = dist[0]

	#This sorts the distances (makes it easer to work with in the future)
	#sorted() returns items in an ascending or descending list 
	#operator.itemgetter() retrieves a specified value from a list
	sorted_d = sorted(distances.items(), key = operator.itemgetter(1))

	#This is an empty list we use to store the neighbours of a point we test
	neighbours = []

	#Extract the top k neighbours
	for x in range(k):
		#Adds the first index of the sorted distances to the neighbours list
		neighbours.append(sorted_d[x] [0])

	#This finds out the class of each neighbour
	class_counter = {}
	for x in range(len(neighbours)):
		#Reminder: [-1] = last index
		cls = training_points.iloc[neighbours[x]][-1]

		#Decide whether or not to give a point to the class based on the value of the neighbour
		if cls in class_counter:
			class_counter[cls] += 1
		else:
			class_counter[cls] = 1

	#Sort the class counter in descending order
	sorted_counter = sorted(class_counter.items(), key = operator.itemgetter(1), reverse = True)

	#Return the class with the most count as well as neighbours as found
	return(sorted_counter[0] [0], neighbours)

#This is the test point used in this example
test_set = [[3, 3.9]]
test = pandas.DataFrame(test_set)
cls, neighbours = knn(data, test, 5)
print("Predicted class: " + cls)


#The following section of code is purely for data visualisation reasons

#If the class is A, make it red, otherwise make the point blue
colours = ['r' if i == 'A' else 'b' for i in data['Class (A or B)']]
#Generate the a scatter plot with axis names x and y, using the colours declared in line 106
ax = data.plot(kind = 'scatter', x = 'x', y = 'y', c = colours)
#.lim() limits the length of axis
matplotlib.pyplot.xlim(0,7)
matplotlib.pyplot.ylim(0,7)

#Plot the test point
#Plot points (0,0) and (0,1), based from the y origin
matplotlib.pyplot.plot(test_set[0][0], test_set[0][1], "yo", markersize = '9')

#This sets the k value to go down in intervals of 2 (from 7 to 1)
for k in range(7,0,-2):
	cls,neighbours = knn(data,test,k)
	print("===========")
	print("k = ", k)
	print("Class ", cls)
	print("neighbours")
	print(data.iloc[neighbours])

	#From the neighbours list, grab the last index value
	furthest_point = data.iloc[neighbours].tail(1)

	#Draw a circle that touches both the test point and the furthest point (for k)
	radius = euclidean_distance(test, furthest_point.iloc[0], 2)

	#Display the circle in red if classification is A, otherwise display it as blue
	c = 'r' if cls == 'A' else 'b'
	#NOTE!!!!!! COLOR, NOT COLOUR HERE!!!!!!!
	circle = matplotlib.pyplot.Circle((test_set[0][0], test_set[0][1]), radius, color = c, alpha = 0.3) 
	ax.add_patch(circle)

matplotlib.pyplot.gca().set_aspect('equal', adjustable = 'box')
matplotlib.pyplot.show()


