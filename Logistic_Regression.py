
#import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn import metrics

dataset = pd.read_csv('survey lung cancer.csv')

dataset['LUNG_CANCER']=dataset['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
x = dataset.iloc[:,1:15].values
y= dataset.iloc[:,-1].values
print(x)

# Split the data into Training and Testing set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

print(y_pred)

# Making the confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print(cm)

print(sum(y_test))

import numpy as np
from sklearn.metrics import accuracy_score
print('accuracy')
print(accuracy_score(y_test, y_pred))

TN = cm [0,0]
FP = cm [0,1]
TP = cm [1,1]
FN = cm [1,0]


''' Plotting ROC Curve '''

y_pred = classifier.predict_proba(x_test)[::,1]
FP, TP, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(FP,TP,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

''' ----------------------  '''


sensetivity = TP/(TP+FP)  
speciticity = TN/(TN+FP)   
false_neg_rate=FN/(FN+TP)

print (sensetivity)
print (speciticity)
print (false_neg_rate)

print("________________")
print(TN)
print(FP)
print(TP)
print(FN)
print("________________")