# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 10:50:19 2026

@author: sowja
"""
#STEP 1 :import lib
import pandas as pd
import numpy as np

#step 2 - load daataset
df=pd.read_csv("D:\GQT Internship data science\loan_data.csv")
df.head(5)

#step 3: feayure selection and target

x=df[['age','income','credit_score','loan_amount']]
y=df['approval_status']
 
#step 4: train and split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)

#step 5:train logistic regreession model
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression(max_iter=1000)
log_model.fit(x_train,y_train)

# step 6: Prediction and evaluation
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred_log=log_model.predict(x_test)
print("Confusion matrix:\n",confusion_matrix(y_test, y_pred_log))
print("\n classification report:\n",classification_report(y_test, y_pred_log))
print("accuracy:",accuracy_score(y_test, y_pred_log))

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train )
x_test_scaled=scaler.transform(x_test)

#k=3
from sklearn.neighbors import KNeighborsClassifier
knn3=KNeighborsClassifier(n_neighbors=3)
knn3.fit(x_train_scaled,y_train)

y_pred_knn3=knn3.predict(x_test_scaled)
print("K=3 Accuracy",accuracy_score(y_test,y_pred_knn3))

#k=5
from sklearn.neighbors import KNeighborsClassifier
knn5=KNeighborsClassifier(n_neighbors=5)
knn5.fit(x_train_scaled,y_train)

y_pred_knn5=knn5.predict(x_test_scaled)
print("K=5 Accuracy",accuracy_score(y_test,y_pred_knn5))

#k=7
from sklearn.neighbors import KNeighborsClassifier
knn7=KNeighborsClassifier(n_neighbors=7)
knn7.fit(x_train_scaled,y_train)

y_pred_knn7=knn7.predict(x_test_scaled)
print("K=7 Accuracy",accuracy_score(y_test,y_pred_knn7))

from sklearn.tree import DecisionTreeClassifier
dt_model=DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train)

y_pred_dt= dt_model.predict(x_test)
print("Confusion matrix:\n",confusion_matrix(y_test, y_pred_dt))
print("\n classification report:\n",classification_report(y_test, y_pred_dt))
print("accuracy:",accuracy_score(y_test, y_pred_dt))

results=pd.DataFrame({
    'Model':['Logistics Regression','KNN (best K)','Decision Tree'],
    'Accuracy':[
        accuracy_score(y_test, y_pred_log),
        max(
            accuracy_score(y_test,y_pred_knn3),
            accuracy_score(y_test,y_pred_knn5),
            accuracy_score(y_test,y_pred_knn7)
        ),
        accuracy_score(y_test,y_pred_dt)
        ]
    })
print(results)

#df_test=x_test.copy()
#df_test['actual']=y_test
#df_test['predicted']=y_pred_dt
#df_test.to_csv("loan_predictions.csv",index=False)

#logical Regression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#import numpy as np
cm_log= confusion_matrix(y_test, y_pred_log)

plt.figure()
plt.imshow(cm_log)
plt.title("confusion matrix - Logistics regression")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.xticks([0,1])
plt.yticks([0,1])

for i in range (2):
    for j in range (2):
        plt.text(j,i,cm_log[i,j],ha="center",va="center")
plt.show()