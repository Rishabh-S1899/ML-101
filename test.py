import pandas as pd
from sklearn.linear_model import LogisticRegression


data = {'Numeric1': [1, 2, 3, 4, 5],
        'Numeric2': [6, 7, 8, 9, 10],
        'Gender': ['male', 'female', 'male', None, None]}

df = pd.DataFrame(data)
print(df)
df_missing = df[df['Gender'].isna()]
df_not_missing = df[~df['Gender'].isna()]
print(df_not_missing)
logistic_regression = LogisticRegression()

# Fit the logistic regression model using the numerical columns to predict "Gender"
logistic_regression.fit(df_not_missing[['Numeric1', 'Numeric2']], df_not_missing['Gender'])

predicted_values = logistic_regression.predict(df_missing[['Numeric1', 'Numeric2']])

# Replace the missing "Gender" values with the predicted values
df.loc[df['Gender'].isna(), 'Gender'] = predicted_values


print(df)