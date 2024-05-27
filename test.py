import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data directly from the CSV file
companies = pd.read_csv('1000_Companies.csv', delimiter='\t')
print(companies.head())

# Exploratory Data Analysis (EDA)
print(companies.info())
print(companies.describe())
sns.pairplot(companies)
plt.show()

# Data Preprocessing
companies = pd.get_dummies(companies, columns=['State'], drop_first=True)
print(companies.head())

# Splitting the data
X = companies.drop('Profit', axis=1)
y = companies['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualizing the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title('Actual vs Predicted Profit')
plt.show()
