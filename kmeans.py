import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

data={"customer":["Riya","Aman","Faizan","Neha","Imran","Sneha"],
      "age":[20,30,40,22,38,25],
      "expending":[100,200,300,110,290,130]
      }
df=pd.DataFrame(data)
x=df[["age","expending"]]
model=KMeans(n_clusters=2,random_state=42,n_init=10)

df["group"]=model.fit_predict(x)
plt.figure(figsize=(6,5))

for group in df["group"].unique(): #[0,1]
    group_data=df[df["group"]==group]
    plt.scatter(group_data["age"],group_data["expending"],label=f"group{group}")

plt.xlabel("age")
plt.ylabel("expending")
plt.title("customre segments (k-means)")
plt.legend()
plt.grid(True)
plt.show()
print(df)


