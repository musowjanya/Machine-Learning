import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
df=pd.read_csv("D:\GQT Internship data science\housing_prices.csv")
df.head()
df.tail()
df.shape
df.columns #to see name of all columns
df.info() #information about dataset
df.describe()
df.isnull().sum()

plt.figure()
plt.hist(df['Price_Lakhs'],bins=20) 
plt.title("price distribution")
plt.xlabel("price")
plt.ylabel("frequency")
plt.show()

plt.figure()
plt.scatter(df['Area_sqft'],df['Price_Lakhs'])
plt.xlabel("area")
plt.ylabel("price")
plt.title("area vs price")
plt.show()

df.groupby('Location')['Price_Lakhs'].mean().plot(kind='bar')
plt.title("average price by location ")
plt.xlabel("location")
plt.ylabel("average price")
plt.show()

corr=df.corr(numeric_only=True)
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True)
plt.title("corelation hestmap")
plt.show()

df.groupby('Bedrooms')['Price_Lakhs'].mean().plot(kind='bar')
plt.title("average price by number of beedrooms ")
plt.xlabel("beedrooms")
plt.ylabel("average price")
plt.show()

df.groupby('Area_sqft')['Price_Lakhs'].mean().plot(kind='bar')
plt.title("average price by area ")
plt.xlabel("Area")
plt.ylabel("average price")
plt.show()

df.to_csv("housing_eda_ready.csv", index=False)

