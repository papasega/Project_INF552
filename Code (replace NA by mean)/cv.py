import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from Dung_gini import gini

def cv(inv_penal,v,k):
    # Return the cross-validation gini of logistics regression with parameters inv_penal,v,k (where inv_penal is the C (inverse penalization strength) of the logistic regression model)

    # Read data
    train = pd.read_csv('../Data/train.csv')
    # Create cross-validation sets
    alpha = 0.6 # Using alpha of train_0 and train_1 to train, 1-alpha to test
    train_0 = train.query('target==0')
    train_1 = train.query('target==1')
    indices_0_train = np.random.choice(len(train_0),size=int(alpha*len(train_0)),replace=False)
    indices_1_train = np.random.choice(len(train_1),size=int(alpha*len(train_1)),replace=False)
    indices_0_test = np.delete(np.arange(len(train_0)),indices_0_train)
    indices_1_test = np.delete(np.arange(len(train_1)),indices_1_train)
    train_cv = pd.concat([train_0.iloc[indices_0_train,:],train_1.iloc[indices_1_train,:]],axis=0)
    test_cv = pd.concat([train_0.iloc[indices_0_test,:],train_1.iloc[indices_1_test,:]],axis=0)
    train=train_cv
    test=test_cv
    # Throw away unnecessary variables
    train_id = train['id'].values
    train_target = train['target'].values
    test_id = test['id'].values
    test_id = test_id.astype(int)
    train = train.drop('target',axis=1)
    train = train.drop('id',axis=1)
    test = test.drop('id',axis=1) 
    test_target = test['target'].values
    test = test.drop('target',axis=1) #train and test are no longer changed from now

    # Separating num and cat in train and test
    heap = pd.concat([train,test],axis=0)
    cat_columns = [col for col in heap.columns if (('bin' in col)|('cat' in col))]
    num_columns = [col for col in heap.columns if (('bin' not in col) and ('cat' not in col) and (col!='id') and (col!='target'))]
    heap_num = heap[num_columns].values
    heap_cat = heap[cat_columns]
    heap_cat = pd.get_dummies(heap_cat,columns=heap_cat.columns,drop_first=True).values
    train_num = heap_num[0:len(train),:]
    train_cat = heap_cat[0:len(train),:]
    test_num = heap_num[len(train):,:]
    test_cat = heap_cat[len(train):,:]

    # Imputing missing values
    imp = Imputer(missing_values=-1,strategy='mean',axis=0)
    imp.fit(train_num)
    train_num = imp.transform(train_num)
    test_num = imp.transform(test_num)

    # Scaling data
    mean_train_num = np.mean(train_num,axis=0)
    sd_train_num = np.std(train_num,axis=0)
    train_num = (train_num - mean_train_num)/sd_train_num
    test_num = (test_num - mean_train_num)/sd_train_num
    mean_train_cat = np.mean(train_cat,axis=0)
    sd_train_cat = np.std(train_cat,axis=0)
    train_cat = (train_cat - mean_train_cat)/sd_train_cat
    test_cat = (test_cat - mean_train_cat)/sd_train_cat

    # Choosing v numerical variables most correlated with the target in the training set, then polynomialize them to order k
    train_target_scl = (train_target-np.mean(train_target))/np.std(train_target)
    fea_cov = np.abs(train_target_scl.dot(train_num)) #fea_cov[i] is abs of the not-divided-by-n covariance between the scaled target
    # and the feature [i] of the training set
    imp_fea = np.argsort(-fea_cov)[0:v] #the v most important features
    heap_num = np.concatenate((train_num,test_num),axis=0)
    heap_num_1 = heap_num[:,imp_fea] # we will polynomialize this matrix
    heap_num_0 = np.delete(heap_num,imp_fea,axis=1)
    # Start polynomializing
    poly = PolynomialFeatures(degree=k,include_bias=False)
    heap_num_1 = poly.fit_transform(heap_num_1)
    # Reseparate
    heap_num = np.concatenate((heap_num_0,heap_num_1),axis=1)
    train_num = heap_num[0:len(train),:]
    test_num = heap_num[len(train):,:]

    # Give out the processed matrices
    train_es = np.concatenate((train_num,train_cat),axis=1) #esential of the training matrix
    test_es = np.concatenate((test_num,test_cat),axis=1)

    # Start learning
    #print("Kernelling done. Start learning.")
    model = LogisticRegression(C=inv_penal)
    model.fit(train_es,train_target)
    prediction = model.predict_proba(test_es)
    assert(prediction.shape[0]==len(test_id))
    assert(prediction.shape[1]==2)
    assert(prediction[0,0]>1/2)
    prediction = prediction[:,1]

    return gini(test_target,prediction)
