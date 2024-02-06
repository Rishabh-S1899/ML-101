import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as np



df=pd.read_csv(r'C:\Users\risha\Desktop\ML 101\ML101_train_dataset.csv')
print(df.columns)

df_interpolate=df[['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age','Gender', 'Height', 'Weight', 'Cholesterol level', 'Blood Sugar level']].copy()

columns_to_replace_nan = [ 'Height', 'Weight', 'Cholesterol level', 'Blood Sugar level']
# Iterate through each column and replace NaN with the mean of that column
for column in columns_to_replace_nan:
    mean_value = df_interpolate[column].mean()
    df_interpolate[column].fillna(mean_value, inplace=True)
# print(df_interpolate.isnull().sum())

columns_to_interpolate = ['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age']
df_interpolate[columns_to_interpolate] = df_interpolate[columns_to_interpolate].interpolate(method='linear')

df_missing = df_interpolate[df_interpolate['Gender'].isna()]
df_not_missing = df_interpolate[~df_interpolate['Gender'].isna()]
logistic_regression = LogisticRegression()
# print('this is df_not_missing')
# print(df_missing)
# print('this is df_not_missing')
# print(df_not_missing)
logistic_regression.fit(df_not_missing[['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age','Height', 'Weight', 'Cholesterol level', 'Blood Sugar level']], df_not_missing['Gender'])

predicted_values = logistic_regression.predict(df_missing[['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age','Height', 'Weight', 'Cholesterol level', 'Blood Sugar level']])

# Replace the missing "Gender" values with the predicted values
df_interpolate.loc[df_interpolate['Gender'].isna(), 'Gender'] = predicted_values

print(df_interpolate)
plt.hist(df_interpolate['Gender'])
# plt.show()
df_interpolate['Systolic BP']=df['Systolic BP']
df_interpolate['Diastolic BP']=df['Diastolic BP']
df_interpolate['LifeStyle']=df['LifeStyle']

# print(df_interpolate)


def detect_outliers(column, threshold=1.5):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return outliers.tolist()

# Create a dictionary to store outliers
outliers_dict = {}

# Iterate through the columns and detect outliers
for column_name in ['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age']:
    outliers_dict[column_name] = detect_outliers(df_interpolate[column_name])

print("Outliers Dictionary:")
print(outliers_dict)

# # Define the outlier detection function using Z-scores
# def remove_outliers_zscore(df, threshold=3):
#     z_scores = np.abs(stats.zscore(df))
#     mask = (z_scores <= threshold).all(axis=1)
#     return df[mask]

# # Remove rows with outliers in each column using Z-scores
# threshold = 3  # Adjust this threshold as needed
# df_no_outliers = remove_outliers_zscore(df_interpolate, threshold)

# print("DataFrame with Outliers Removed (Z-score method):")
# print(df_no_outliers)


def remove_outliers(dataframe, columns, z_threshold=3):
    outlier_indices = set()
    
    for column in columns:
        z_scores = np.abs(stats.zscore(dataframe[column]))
        outlier_indices.update(set(np.where(z_scores > z_threshold)[0]))

    # Remove duplicate indices and sort them
    outlier_indices = list(outlier_indices)
    outlier_indices.sort()

    # Create a new DataFrame without the outlier rows
    cleaned_dataframe = dataframe.drop(outlier_indices)

    return cleaned_dataframe


df_clean=remove_outliers(df_interpolate, [ 'Height', 'Weight', 'Cholesterol level', 'Blood Sugar level'], z_threshold=3)

print(df_clean)


df_diastol=df_interpolate[['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age','Gender','Height', 'Weight', 'Cholesterol level', 'Blood Sugar level','Diastolic BP']]
df_systol=df_interpolate[['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age','Gender','Height', 'Weight', 'Cholesterol level', 'Blood Sugar level','Systolic BP']]
df_lifestyle=df_interpolate[['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age','Gender','Height', 'Weight', 'Cholesterol level', 'Blood Sugar level','LifeStyle']]

df_systol.to_csv('Systol.csv')
df_diastol.to_csv('diastol.csv')
df_lifestyle.to_csv('LifeStyle.csv')


# df_interpolate.to_csv('ML101_Train_corrected.csv')


#Outlier 






























# df_interpolate.isnull().sum()
# columns_to_interpolate_KNN = ['Average Daily Steps', 'Hours of Sleep', 'Caloric Intake', 'Age', 'Height', 'Weight', 'Cholesterol level', 'Blood Sugar level']

# X=df_interpolate[columns_to_interpolate_KNN]
# imputer = IterativeImputer(max_iter=1000, random_state=0)
# imputer.fit(X)
# df_imputed=imputer.transform(X)
# df_corrected=pd.DataFrame(df_imputed)

# df_corrected

# x=[]
# for i in df_corrected.index:
#   x.append(i)
#   l=df_corrected[2]
# plt.bar(x,l)
# plt.show()

