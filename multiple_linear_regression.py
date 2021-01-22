"""Exploring unsupervised learning using
multiple linear regression"""
"""useful when many variables are involved
and somehow linked in a linear manner."""
# import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset
dataset = pd.read_csv("data/50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# view the data
dataset.head()
# import libraries which aid in processing qualitative data into quantitative
# By transforming categorical variables into numeric ones(dummy variables):
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder = LabelEncoder()
# Transform the qualitative data at row[3] == 'State' 
X[:, 3] = labelencoder.fit_transform(X[:,3]) 

onehotencoder =  OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Divide data into Training, Test sets,
# Create regressor and the fit line.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction
y_pred = regressor.predict(X_test)

print(f"Predictions: {y_pred}")
print(f"Y Test Output: {y_test}")






 










