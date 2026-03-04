import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
"customer":["Riya","Aman","Faizan","Neha","Imran","Sneha"],
"Spending":[20,30,42,22,38,25],
"Visits":[10,20,30,11,29,13]        
 }

df = pd.DataFrame(data)

x = df[['Spending','Visits']]

kmeans = KMeans(n_clusters = 2,random_state=42)
df['Cluster'] = kmeans.fit_predict(x)

plt.scatter(df['Spending'],df['Visits'],c=df['Cluster'])
plt.xlabel("Spending")
plt.ylabel("Visits")
plt.grid(True)
plt.show()
