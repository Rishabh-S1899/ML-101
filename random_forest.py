import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score  # for classification
from sklearn.metrics import mean_squared_error  # for regression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import numpy as np


from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_classification

df_cleaned = pd.read_csv('Systol_corrected.csv')


df = df_cleaned.copy()

df = df[['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age',      
       'Gender', 'Height', 'Weight', 'Cholesterol level', 'Blood Sugar level',
       'Systolic BP']]

target_column = 'Systolic BP'

# Calculate the IQR for the target column
Q1 = df[target_column].quantile(0.25)
Q3 = df[target_column].quantile(0.75)
IQR = Q3 - Q1

# Define a threshold for outliers
threshold = 0.005

# Create a boolean mask to identify outliers
outliers_mask = (df[target_column] < (Q1 - threshold * IQR)) | (df[target_column] > (Q3 + threshold * IQR))

# Remove rows with outliers
df_cleaned = df[~outliers_mask]
# data = pd.read_csv(r'C:\Users\risha\Desktop\ML 101\Systol_corrected.csv')
# data.drop(['Unnamed'])


X = df_cleaned[['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age','Gender','Height', 'Weight', 'Cholesterol level', 'Blood Sugar level']]
# print(X)
y = df_cleaned['Systolic BP']
# print(y)
# If 'gender' is categorical, one-hot encode it

X = pd.get_dummies(X, columns=['Gender'], drop_first=True)
# print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(y_train,y_test)
# Create and define the Random Forest model
# For classification:
# model = RandomForestClassifier()

# # For regression:
# model = RandomForestRegressor()

# # Define hyperparameter ranges for RandomizedSearchCV
# param_dist = {
#     'n_estimators': randint(10, 200),
#     'max_depth': randint(1, 20),
#     'min_samples_split': uniform(0, 1),
# }
# # Create a RandomizedSearchCV object
# random_search = RandomizedSearchCV(
#     model, param_distributions=param_dist, n_iter=1, cv=25, n_jobs=-1)

# # Fit the RandomizedSearchCV to the training data
# random_search.fit(X_train, y_train)

# # Get the best model with tuned hyperparameters
# best_model = random_search.best_estimator_

# # Make predictions with the best model
# y_pred = best_model.predict(X_test)

# for i in range(len(list(y_pred))):
#     print(list(y_pred)[i],list(y_test)[i])

# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')

# # Print the best hyperparameters found
# print("Best Hyperparameters:", random_search.best_params_)

# from sklearn.linear_model import Ridge

# # Create the Ridge Regression model
# model = Ridge()

# # Define hyperparameter ranges for RandomizedSearchCV
# param_dist = {
#     'alpha': uniform(0, 1)  # Range for the ridge penalty hyperparameter alpha
# }

# # Create a RandomizedSearchCV object
# random_search = RandomizedSearchCV(
#     model, param_distributions=param_dist, n_iter=1, cv=100, n_jobs=-1,scoring='neg_mean_squared_error'
# )

# # Fit the RandomizedSearchCV to the training data
# random_search.fit(X_train, y_train)

# # Get the best model with tuned hyperparameters
# best_model = random_search.best_estimator_

# # Make predictions with the best model
# y_pred = best_model.predict(X_test)

# # Evaluate the best model using Mean Squared Error (MSE)
# mse = mean_squared_error(y_test, y_pred)

# for i in range(len(list(y_pred))):
#     print(list(y_pred)[i],list(y_test)[i])
# print(f'Mean Squared Error: {mse}')

# # Print the best hyperparameters found
# print("Best Hyperparameters:", random_search.best_params_)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset
# data = pd.read_csv('your_data.csv')

# Extract the independent variables (features) and the target variable
X_numerical = df_cleaned[['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age','Height', 'Weight', 'Cholesterol level', 'Blood Sugar level']]
X_categorical = pd.get_dummies(df_cleaned['Gender'], drop_first=True)  # One-hot encoding for the 'gender' variable
X = pd.concat([X_numerical, X_categorical], axis=1)
y = df_cleaned['Systolic BP']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Multiple Linear Regression model
model = LinearRegression()

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)

print("y test: ",y_test,'y pred: ',y_pred)
print(f'Mean Squared Error: {mse}')
