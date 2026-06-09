import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
data = {
    "student_id": [101,102,103,104,105,106,107,108,109,110],
    "student_name": ["Aman","Riya","Kunal","Neha","Aman","Rohit","Priya","Ankit","Riya","Simran"],
    "age": [20,21,19,22,20,23,21,20,19,22],
    "height": [165,170,155,160,300,172,168,158,162,159],  # 300 is outlier
    "weight": [55,60,50,58,200,65,59,54,52,57],  # 200 is outlier
    "semester": [3,4,2,5,3,6,4,3,2,5],
    "gender": ["M","F","M","F","M","M","F","M","F","F"],
    "city": ["Delhi","Mumbai","Delhi","Pune","Delhi","Chennai","Mumbai","Delhi","Pune","Mumbai"]
}

df = pd.DataFrame(data)
print(df)
print("shape of dataset:",df.shape)
print("column names:",df.columns)
print("null:",df.isnull().sum())
print("unique student names:")
print(df["student_name"].unique())
plt.figure()
plt.plot(df["height"],df["weight"])
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Height vs Weight")
plt.show()
plt.figure()
plt.plot(df["height"],df["age"])
plt.xlabel("height")
plt.ylabel("age")
plt.title("height vs age")
plt.show()
plt.figure()
plt.plot(df["weight"],df["age"])
plt.xlabel('weight')
plt.ylabel("age")
plt.show()
plt.figure()
sns.scatterplot(x=df.index,y=df["height"])
plt.title("scatterplot for height")
plt.show()
plt.figure()
sns.boxplot(data=df)
plt.title("boxplot for all columns")
plt.show()
Q1=df["height"].quantile(0.25)
Q3=df["height"].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
df_no_height_outlier=df[(df["height"]>=lower_bound)&(df["height"]<=upper_bound)]
print("DATASET AFTER REMOVING HEIGHT OUTLIERS:")
print(df_no_height_outlier)
z_scores=np.abs(stats.zscore(df_no_height_outlier["weight"]))
df_final=df_no_height_outlier[z_scores<3]
print("final dataset after removing weight outliers:")
print(df_final)