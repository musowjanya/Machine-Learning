import pandas as pd #data manupiation
import numpy as np #mathematical computation

df=pd.read_csv("D:\GQT Internship data science\loan_data.csv")

x=df[['age','income','credit_score','loan_amount']]
y=df['approval_status']

from sklearn.model_selection import train_test_split #split th data to train and test data 
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rf_model= RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train,y_train)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
y_pred_rf=rf_model.predict(x_test)
print("confusion Matrix\n",confusion_matrix(y_test,y_pred_rf))
print("classificatio report:\n",classification_report(y_test,y_pred_rf))
print("accuracy_score",accuracy_score(y_test,y_pred_rf))

#FEATURE IMPORTANCE
importance=pd.DataFrame({'feature':x.columns, 'Importance':rf_model.feature_importances_}).sort_values(by='Importance',ascending=False) ##

#SUPPORT VECTOR MACHINE
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler=scaler.transform(x_test)

from sklearn.svm import SVC
svm_model=SVC(kernel='rbf')
svm_model.fit(x_train_scaled, y_train)

y_pred_svm=svm_model.predict(x_test_scaled)
print("confusion Matrix\n",confusion_matrix(y_test,y_pred_svm))
print("classificatio report:\n",classification_report(y_test,y_pred_svm))
print("accuracy_score",accuracy_score(y_test,y_pred_svm))

print("random foredt accuracy", accuracy_score(y_test, y_pred_rf))
print("SVM accuracy:",accuracy_score(y_test,y_pred_svm))

