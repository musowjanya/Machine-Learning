import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("D:\GQT Internship data science\quikr_car.csv")
#df.head()
#df.info()
#print(dir(pd))
#df.isnull()
#df.isnull().sum()
#print(df["name"])
#print(df["name"].str.split())
#print(df['name'].str.split().str.get(0))
#print(df['name'].str.split().str.slice(0,3))
#df['name']=df['name'].str.split().str.slice(0,3)
#df['name']=df['name'].str.join(" ")
#print(df['name'])
#print(df['Price']!="Ask For Price")
#df[df['Price']!="Ask For Price"]
#df=df[df['Price']!="Ask For Price"]
#print(df)

#print(df["Price"].str.replace(",",""))
#df["Price"]=df["Price"].str.replace(",","")

#df=df[df["kms_driven"]!="petrol"]
#print(df)
#print(df["kms_driven"].isnull().sum())

#print(df["kms_driven"].fillna(0,inplace=True))
#df["kms_driven"]=df["kms_driven"].str.replace(","," ")
#print(df["kms_driven"])

x=df["company"].head(10)
y=df["Price"].head(10)
plt.bar(x,y)
plt.xlabel("company")
plt.ylabel("price")
plt.title("bar graph")

#x=df["company"].head(10)
#y=df["Price"].head(10)
#plt.pie(y)
#plt.ylabel("price")
#plt.title("pie chart")

h=df["company"].value_counts()
x=h.keys()
y=h
plt.figure(figsize=(25,10))
plt.bar(x,y)
plt.xticks(rotation=40)
plt.show()

plt.figure(figsize=(12,10))
sns.boxplot(x="company",y="Price",data=df,color="purple")
plt.xticks(rotation=40)
plt.show()
