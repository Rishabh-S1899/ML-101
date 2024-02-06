import pandas as pd



df_systol=pd.read_csv('Systol.csv')
df_diastol=pd.read_csv('diastol.csv')
df_lifestyle=pd.read_csv('LifeStyle.csv')


df_systol=df_systol.dropna()
df_diastol=df_diastol.dropna()
df_lifestyle=df_lifestyle.dropna()


df_systol.to_csv('Systol_corrected.csv',index=False)
df_diastol.to_csv('diastol_corrected.csv',index=False)
df_lifestyle.to_csv('LifeStyle_corrected.csv',index=False)