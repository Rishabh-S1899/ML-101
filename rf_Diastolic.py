import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score  # for classification
from sklearn.metrics import mean_squared_error  # for regression
import matplotlib.pyplot as plt



df_test = pd.read_csv("test.csv")
# Load your dataset
# Replace 'your_data.csv' with the path to your dataset
data = pd.read_csv('diastol_corrected.csv')

# Split your dataset into features (X) and the target variable (y)
data= data[['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age',      
       'Gender', 'Height', 'Weight', 'Cholesterol level', 'Blood Sugar level',
       'Diastolic BP']]  # Drop the 10th column (target)
# y = data['Diastolic BP']  # The 10th column (target)
target_column = 'Diastolic BP'
# Calculate the IQR for the target column
Q1 = data[target_column].quantile(0.25)
Q3 = data[target_column].quantile(0.75)
IQR = Q3 - Q1

# Define a threshold for outliers
threshold = 1.15

# Create a boolean mask to identify outliers
outliers_mask = (data[target_column] < (Q1 - threshold * IQR)) | (data[target_column] > (Q3 + threshold * IQR))

# # Remove rows with outliers
df_cleaned = data[~outliers_mask]
# data = pd.read_csv(r'C:\Users\risha\Desktop\ML 101\Systol_corrected.csv')
# data.drop(['Unnamed'])

# df_cleaned=data
X = df_cleaned[['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age','Gender','Height', 'Weight', 'Cholesterol level', 'Blood Sugar level']]
# print(X)
y = df_cleaned['Diastolic BP']

# plt.hist(y)
# plt.show()
# If 'gender' is categorical (e.g., 'Male' and 'Female'), you need to one-hot encode it
X = pd.get_dummies(X, columns=['Gender'], drop_first=True)  # Drop the first category to avoid multicollinearity

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# For regression:
model = RandomForestRegressor(n_estimators=300, random_state=42)

model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# For regression:
mse = mean_squared_error(y_test, y_pred)
for i in range(len(list(y_pred))):
    print(list(y_pred)[i],list(y_test)[i])
print(f'Mean Squared Error: {mse}')

df_test["Diastolic BP"] = y_pred

df_test.to_csv("final_dystolic_test.csv", index=False)