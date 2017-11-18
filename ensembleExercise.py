# coding: utf-8

# https://www.python.org/dev/peps/pep-0008#introduction<BR>
# http://scikit-learn.org/<BR>
# http://pandas.pydata.org/<BR>

#%%
import numpy as np
import pandas as pd
import pylab as plt

### Fetch the data and load it in pandas
data = pd.read_csv('train.csv')
#data = data.sample(10000)

#print("Size of the data: ", data.shape)

#%%
# See data (five rows) using pandas tools
#print(data.head())


### Prepare input to scikit and train and test cut

'''
binary_data = data[np.logical_or(data['Cover_Type'] == 1,data['Cover_Type'] == 2)] # two-class classification set
X = binary_data.drop('Cover_Type', axis=1).values
y = binary_data['Cover_Type'].values
#print(np.unique(y)) # in cac loai label
y = 2 * y - 3 # converting labels from [1,2] to [-1,1]
'''
X = data.drop('target',axis=1)
X_train = X.drop('id',axis=1).values
#X = X.drop('id',axis=1).values
y = data['target'].values
y_train=2*y-1
#y = 2*y-1

#%%
# Import cross validation tools from scikit
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
test = pd.read_csv('test.csv')
#test = test.sample(10000)
X_test = test.drop('id',axis=1).values
ids= test['id']
#%%
### Train a single decision tree

from sklearn.tree import DecisionTreeClassifier
'''
clf = DecisionTreeClassifier(max_depth=8)

# Train the classifier and print training time #o dau ? 
clf.fit(X_train, y_train)

#%%
# Do classification on the test dataset and print classification results
from sklearn.metrics import classification_report
target_names = data['Cover_Type'].unique().astype(str).sort()
y_pred = clf.predict(X_test)
#print(classification_report(y_test, y_pred, target_names=target_names))
# in report cua classification
'''
#%%
# Compute accuracy of the classifier (correctly classified instances)
from sklearn.metrics import accuracy_score
#accuracy_score(y_test, y_pred)



#===================================================================
#%%
### Train AdaBoost

# Your first exercise is to program AdaBoost.
# You can call *DecisionTreeClassifier* as above, 
# but you have to figure out how to pass the weight vector (for weighted classification) 
# to the *fit* function using the help pages of scikit-learn. At the end of 
# the loop, compute the training and test errors so the last section of the code can 
# plot the lerning curves. 
# 
# Once the code is finished, play around with the hyperparameters (D and T), 
# and try to understand what is happening.

D = 50# tree depth #15  
T = 1000 #1000 # number of trees #200
w = np.ones(X_train.shape[0])+10000*(y_train+1)/2 
w=w/sum(w)
#training_scores = np.zeros(X_train.shape[0])
#test_scores = np.zeros(X_test.shape[0])

#ts = plt.arange(len(training_scores))
training_errors = []
test_errors = []


#===============================
# Your code should go here
yp=np.zeros((X_train.shape[0],T)) #train prediction
ytest=np.zeros((X_test.shape[0],T)) #test prediction
alpha=np.zeros(T)
for t in range(T):
    #find yt
    clf = DecisionTreeClassifier(max_depth=D)
    clf.fit(X_train, y_train, sample_weight=w)
    yt=clf.predict(X_train)
    #save in yp
    yp[:,t]=yt
    #for the test
    ytest[:,t]=clf.predict(X_test)
    #print(yt)
    wrong_predict=1-(yt==y_train)*1
    #a=w*wrong_predict
    #training_scores[t]
    gammat=sum(w*wrong_predict)/sum(w)
    alpha[t]=np.log(1/gammat-1)
    #for i in range(len(w)):
    #    w[i]=w[i]*np.exp(alpha[t]*wrong_predict[i])
    w=w*np.exp(alpha[t]*wrong_predict)

trainprediction=(np.dot(yp,alpha)>0)*2-1
rate_train=accuracy_score(y_train,trainprediction)
#true_predict=(trainprediction==y_train)*1
#rate_train=sum(true_predict)/len(w)
'''
testprediction=(np.dot(ytest,alpha)>0)*2-1
#true_predict=(testprediction==y_test)*1
#rate_test=sum(true_predict)/X_test.shape[0]
rate_test1=accuracy_score(y_test,testprediction)
    # Your code should go here
'''

#proba and print
proba=np.dot(ytest,alpha)
proba=(proba-min(proba))/max(1e-10,(max(proba)-min(proba)))    
result=pd.DataFrame(proba,columns=['target'],index=ids)
result.to_csv('sub1.csv',index_label='id')
#===============================

#  Plot training and test error
#plt.plot(training_errors, label="training error")
#plt.plot(test_errors, label="test error")
#plt.legend()



#===================================================================
#%%
### Optimize AdaBoost

# Your final exercise is to optimize the tree depth in AdaBoost. 
# Copy-paste your AdaBoost code into a function, and call it with different tree depths 
# and, for simplicity, with T = 100 iterations (number of trees). Plot the final 
# test error vs the tree depth. Discuss the plot.

#===============================

# Your code should go here
# ten bien giong nhau co sao ko ? 
'''
def TranAdaBoost(X_train,y_train,X_test,y_test,D,T=100):
    yp=np.zeros((X_train.shape[0],T)) #train prediction
    ytest=np.zeros((X_test.shape[0],T)) #test prediction
    w = np.ones(X_train.shape[0]) / X_train.shape[0]    #weight   
    alpha=np.zeros(T)
    for t in range(T):
        #find yt
        clf = DecisionTreeClassifier(max_depth=D)
        clf.fit(X_train, y_train, sample_weight=w)
        yt=clf.predict(X_train) #train prediction at t
        #save in yp
        yp[:,t]=yt
        #for the test
        ytest[:,t]=clf.predict(X_test)
        #print(yt)
        wrong_predict=1-(yt==y_train)*1
        #a=w*wrong_predict
        #training_scores[t]
        gammat=sum(w*wrong_predict)/sum(w)
        alpha[t]=np.log(1/gammat-1)
        for i in range(len(w)):
            w[i]=w[i]*np.exp(alpha[t]*wrong_predict[i])                
    trainprediction=(np.dot(yp,alpha)>0)*2-1
    true_predict=(trainprediction==y_train)*1
    rate_train=sum(true_predict)/X_train.shape[0]    
    testprediction=(np.dot(ytest,alpha)>0)*2-1
    true_predict=(testprediction==y_test)*1
    rate_test=sum(true_predict)/X_test.shape[0]    
    return (1-rate_test), (1-rate_train)


dimension=[] 
for j in range(10): #15,20
    test_E,train_E= TranAdaBoost(X_train,y_train,X_test,y_test,j+1,100)
    test_errors.append(test_E)
    training_errors.append(train_E)
    dimension.append(j+1)
    

plt.plot(dimension,training_errors, label="training error")
plt.plot(dimension,test_errors, label="test error")
plt.legend()
'''
#===============================
    


