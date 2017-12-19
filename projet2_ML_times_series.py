# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 18:13:20 2017

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 01:25:42 2017

@author: hp
"""

import numpy as np
import gc;  gc.enable()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#from xgboost import XGBClassifier

# importation data 

#class Garbage :
#    def __init__(self , i) :
#        self . i = i
#        print(" creating " , self.i)
#    def __del__(self) :
#        print( " ∗∗ deleting" , self.i )
#        
#for i in range(10000) :
#     Garbage(i)
#       
#for g in gc.garbage : # pour chaque objet non−récupérable...
#  g.next = None # on casselecycle . . .
#del gc.garbage[:] # puis on détruit le tout !

train_p2 = pd.read_csv("C:/Users/hp/Desktop/Cours_3A/machnique learning 1/projet2/train.csv")
train_v2_p2 = pd.read_csv("C:/Users/hp/Desktop/Cours_3A/machnique learning 1/projet2/train_v2.csv")
print(train_p2.describe())
print(train_p2.head().T)
#transactions = pd.read_csv("C:/Users/hp/Desktop/Cours_3A/machnique learning 1/projet2/transactions.csv")
members_v3 = pd.read_csv("G:/a_Projet2_Machine_learning/projet2/members_v3.csv")



plt.figure(figsize=(10, 4), dpi = 70)
sns.countplot(train_p2['is_churn'], palette ='rainbow')
plt.xlabel('Is_Churn')
train_p2['is_churn'].value_counts()
###########

train_missing_data = train_p2.isnull().sum().sort_values(ascending=False)
#plt.plot(transactions[1:12])
#plt.show()
##############################################
# Time Series Prediction with LSTM Recurrent Neural Networks in Python with Keras

# 1) LSTM Network for Regression

# fix random seed for reproducibility
np.random.seed(7)

#transactions = transactions.values
#transactions = transactions.astype('float32')

train_p2 = train_p2['msno'].astype(float)

try :
    float(train_p2['msno'])
except ValueError:
    pass 



# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
train_p2 = scaler.fit_transform(train_p2)

# split into train and test sets
train_p2_size = int(len(train_p2) * 0.67)
test_size = len(train_v2_p2) - train_p2_size
train, test = train_p2[0:train_p2_size,:], train_v2_p2[train_p2_size:len(train_v2_p2),:]
print(len(train), len(test))
#
## convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)



# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


# shift train predictions for plotting

trainPredictPlot = np.empty_like(train_p2)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(train_p2)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(train_p2)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(train_p2))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()