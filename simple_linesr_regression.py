import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X_train=pd.read_csv('Linear_X_Train.csv')
X_test=pd.read_csv('Linear_X_Test.csv')
y_train=pd.read_csv('Linear_Y_Train.csv')

#linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

#predicting values
y_pred=regressor.predict(X_test)

#visualising the training set results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title('Score vs Performance_in_Exam (Training Set)')
plt.xlabel('Performance_in_Exam')
plt.ylabel('Score')
plt.show()

#Visualising the Test set results
plt.scatter(X_test, regressor.predict(X_test), color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Score vs Performance_in_Exam (Test Set)')
plt.xlabel('Performance_in_Exam')
plt.ylabel('Score')
plt.show()