#Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
#or u can write it as,
#X = dataset.iloc[:, 1:2].values
# Dont use X = dataset.iloc[:, :1].values because it creates array of 
# vectors same as output y but for these regression models we want 
# metric of input vectors.
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting Linear Regression model to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

y_poly_pred = lin_reg2.predict(poly_reg.fit_transform(X))

#Visualising the Linear Regression Results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


#Visualising the Polynomial Regression Results 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Ploynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# To improve the predictions we can change degree of a polynomial 
#Fitting Polynomial Regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3) #changed from 2 to 3
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

y_poly_pred = lin_reg2.predict(poly_reg.fit_transform(X))

#Visualising the Polynomial Regression Results 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Ploynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
 
# To improve the predictions we can change degree of a polynomial 
#Fitting Polynomial Regression model to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) #changed from 3 to 4
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

y_poly_pred = lin_reg2.predict(poly_reg.fit_transform(X))

#Visualising the Polynomial Regression Results 
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Ploynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#To get more exact curve we can modify our input observation intervals
#Exhere there are 10 observations so there are 10 levels. 1 - 10
# we can form a input metric with more closely spaced input variables to get more accurate curve
#Visualising the Polynomial Regression Results 

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Ploynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Predicting a new result with linear regression
lin_reg.predict(6.5)

#Predicting a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(6.5))










