

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from pandas import DataFrame
from sklearn import linear_model
import statsmodels.api as sm
import pandas as pd

dataset = pd.read_csv('survey lung cancer.csv')



dataset['LUNG_CANCER']=dataset['LUNG_CANCER'].map({'YES': 1, 'NO': 0})

df = DataFrame(dataset,columns=['SMOKING','YELLOW_FINGERS','LUNG_CANCER'])

#print(df)
X = df[['SMOKING','YELLOW_FINGERS']]
Y = df['LUNG_CANCER']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

Predicted_vals = regr.predict(X.values)

print('The R^2 score is: ')
print(r2_score(Y,Predicted_vals))




#prediction with sklearn
SMOKING = 2
YELLOW_FINGERS = 2
print ('Predicted CANCER: \n', regr.predict([[SMOKING ,YELLOW_FINGERS]]))

##
##
## with statsmodels
#X = sm.add_constant(X) # adding a constant
# 
#model = sm.OLS(Y, X).fit()
#predictions = model.predict(X) 
# 
##print_model = model.summary()
#print(print_model)