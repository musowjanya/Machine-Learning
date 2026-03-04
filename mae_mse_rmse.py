import pandas as numpy
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.metrics import mean_absolute_error, mean_squared_error

#real score
real_score=[90,60,80,100]

#model pridict
predicted_score=[85,70,70,95]

mae=mean_absolute_error(real_score,predicted_score)
mse=mean_squared_error(real_score, predicted_score)
rmse=np.sqrt(mse)
print("MAE: on range by:", mae)
print("MSE: Square mistake:",mse)
print("RMSE:final realstic error:",rmse)
