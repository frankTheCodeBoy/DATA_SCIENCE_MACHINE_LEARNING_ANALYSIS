"""Exploring the iris dataset, comparing train and test sets.
"""
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np

# Loading iris dataset
iris = load_iris()

# Array of the data
x = iris.data 
# Array of the labels : answers of each data entry
y = iris.target

# Get label names 
y_names = iris.target_names

# Taking random indices to split the dataset into train and test
test_ids = np.random.permutation(len(x))

# Splitting data and labels into train and test
# Keeping last 10 entries for testing, rest for training
x_train = x[test_ids[:-10]]
x_test = x[test_ids[-10:]]

y_train =  y[test_ids[:-10]]
y_test = y[test_ids[-10:]]

# Classifying using decision tree
clf = tree.DecisionTreeClassifier()

# Training, fitting the classifier with the training set
clf.fit(x_train, y_train)

# Predictions on the test dataset
pred = clf.predict(x_test)

print(pred) # the predicted labels
print(y_test) # actual labels

# prediction accuracy 
result_accuracy = accuracy_score(pred, y_test)
print(float(result_accuracy)*100)



















