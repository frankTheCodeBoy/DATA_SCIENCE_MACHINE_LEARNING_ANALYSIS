"""A demonstration of supervised learning
by simple linear regression."""
# Importing the required libraries.
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv("data/salary.csv")
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , 1].values

# Splitting the dataset into the training set and test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3,
random_state=0)

# Fitting the dataset into Training set and Test set.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set result
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),
color="blue")
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results.
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train),
color="orange")
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()




















