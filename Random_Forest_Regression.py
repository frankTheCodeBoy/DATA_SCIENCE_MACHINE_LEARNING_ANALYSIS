"""Random Forest Regression"""
"""to visualise, analyse non-linear relationships
under supervised learning"""
"""getting results from a collection of decision trees 
generated randomly, averaging results to get best analysis"""
# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the data
dataset = pd.read_csv("data/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting data into training sets, test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Using feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300, random_state=0)
regressor.fit(X,y)

# Prediction of new result
y_pred = regressor.predict(X_test)

# Visualising the Random Forest Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color="red")
plt.plot(X_grid, regressor.predict(X_grid), color="blue")
plt.title("Truth or Bluff (Decision Tree Regressor)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()






















