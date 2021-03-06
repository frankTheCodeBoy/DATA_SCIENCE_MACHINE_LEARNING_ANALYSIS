"""An illustration of supervised learning
featuring linear relationship between variables.
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Get/ Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Utilise only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split data into training and test sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create a linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# Print the coefficient value
print(f"Coefficients: \n {regr.coef_}")

# The mean squared error
print(f"Mean Squared Error: {mean_squared_error(diabetes_y_test, diabetes_y_pred)}")

# Explained variance score: 1 is perfect prediction
print(f"Variance score: {r2_score(diabetes_y_test, diabetes_y_pred)}")

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()






















