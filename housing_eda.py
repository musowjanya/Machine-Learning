import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("D:\GQT Internship data science\housing_eda_ready.csv")
x=df[['Area_sqft']]
y=df['Price_Lakhs']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)

print("MAE",mae)
print("MSE",mse)
print("RMSE",rmse)
print("R2 Score",r2)

plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.xlabel("Area")
plt.ylabel("Simple linear regression") #points should be near the line if it is far then the model is not ok
plt.show()

x=df[['Area_sqft','Bedrooms','Bathrooms']]
y=df['Price_Lakhs']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model_multi=LinearRegression()
model_multi.fit(x_train,y_train)
y_pred_multi=model_multi.predict(x_test)

print("R2 Score", r2_score(y_test, y_pred_multi))
print("RMSE", np.sqrt(mean_squared_error(y_test,y_pred_multi)))

coeff_df=pd.DataFrame(
    model_multi.coef_,x.columns,columns=['Coefficient'])
print(coeff_df)
