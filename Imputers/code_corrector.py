import pandas as pd
import numpy as np
from scipy import stats

df=pd.read_csv(r'C:\Users\risha\Desktop\ML 101\Vamyun\ML101_train_dataset.csv')

df_interpolate=pd.read_csv(r'C:\Users\risha\Desktop\ML 101\Vamyun\final_kNNimputer.csv')


df_interpolate['Systolic BP']=df['Systolic BP']
df_interpolate['Diastolic BP']=df['Diastolic BP']
df_interpolate['LifeStyle']=df['LifeStyle']

print(df_interpolate)


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


df_systol=df_systol.dropna()
df_diastol=df_diastol.dropna()
df_lifestyle=df_lifestyle.dropna()


df_systol.to_csv('Systol_vayun.csv')
df_diastol.to_csv('diastol_vayun.csv')
df_lifestyle.to_csv('LifeStyle_vayun.csv')