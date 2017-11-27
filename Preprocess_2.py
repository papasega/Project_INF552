import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import scale
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LogisticRegression
#from Gini import gini

from sklearn.model_selection import train_test_split



#%% Preprocession
#k = 2; v = 12 #at most 26
"""
Inputs:
train must be obtained by pd.read_csv
We will polynomialize to degree k the v most important numerical variables, if polynome = True
Example: k = 2, v = 10, polynome=True, note: we need v<=26 (or v<26, I'm not sure)
If polynome=False, then k and v can be anything, they have no effect.

Outputs:
List of 4 elements : train, test, train_target, test_id
- train: the processed training matrix, in format numpy, without id and target
- train_target: type numpy
"""
def preprocess(train,test,k=2,v=7,poly=False,to_csv=False):
    # Read data, back up necessary but cubersome variables (id,target) then throw them away
    train_target = train['target'].values
    train = train.drop('target',axis=1)
    train = train.drop('id',axis=1)
    
    test_id = test['id'].values
    test_id = test_id.astype(int)
    test = test.drop('id',axis=1) 
    
    print("Basic reading completed.")
    
    # Separating num and cat in train and test
    len_train=len(train)
    heap = pd.concat([train,test],axis=0)
    del train, test
    
    cat_columns = [col for col in heap.columns if (('bin' in col)|('cat' in col))]
    num_columns = [col for col in heap.columns if (('bin' not in col) and ('cat' not in col) and (col!='id') and (col!='target'))]
    heap_num = heap[num_columns].values
    heap_cat = heap[cat_columns]
    del cat_columns, num_columns, heap
    
    heap_cat = pd.get_dummies(heap_cat,columns=heap_cat.columns,drop_first=True).values
    train_num = heap_num[0:len_train,:]
    train_cat = heap_cat[0:len_train,:]
    test_num = heap_num[len_train:,:]
    test_cat = heap_cat[len_train:,:]
    del heap_cat,heap_num
    
    # Numerical missing values by NA
    train_num[train_num==-1] = np.nan
    test_num[test_num==-1] = np.nan
    
    # Scaling data
    mean_train_num = np.nanmean(train_num,axis=0)
    sd_train_num = np.nanstd(train_num,axis=0)
    train_num = (train_num - mean_train_num)/sd_train_num
    test_num = (test_num - mean_train_num)/sd_train_num
    del mean_train_num, sd_train_num
    
    # for cat
    #mean_train_cat = np.mean(train_cat,axis=0)
    #sd_train_cat = np.std(train_cat,axis=0)
    #train_cat = (train_cat - mean_train_cat)/sd_train_cat
    #test_cat = (test_cat - mean_train_cat)/sd_train_cat
    
    # Replacing all NAs by gaussian
    temp = train_num[np.isnan(train_num)].shape[0]
    train_num[np.isnan(train_num)] = np.random.normal(size=temp)
    temp = test_num[np.isnan(test_num)].shape[0]
    test_num[np.isnan(test_num)] = np.random.normal(size=temp)
    del temp
    print("Basic processing completed.")
    
    #Poly or not :
    if poly==False:
        train = np.column_stack((train_num,train_cat))
        test = np.column_stack((test_num,test_cat))
        if to_csv==True:
            pd.DataFrame(train).to_csv('X_train_ultimate.csv',index=False)
            pd.DataFrame(test).to_csv('X_test_ultimate.csv',index=False)
        return train, test, train_target, test_id
    else:  
        # Choosing v numerical variables most correlated with the target in the training set, then polynomialize them to order k
        train_target_scl = (train_target-np.mean(train_target))/np.std(train_target)
        fea_cov = np.abs(train_target_scl.dot(train_num)) #fea_cov[i] is abs of the not-divided-by-n covariance between the scaled target
        # and the feature [i] of the training set
        imp_fea = np.argsort(-fea_cov)[0:v] #the v most important features
        
        train_num_1 = train_num[:,imp_fea] # we will polynomialize this matrix
        train_num_0 = np.delete(train_num,imp_fea,axis=1)
        test_num_1 = test_num[:,imp_fea] # we will polynomialize this matrix
        test_num_0 = np.delete(test_num,imp_fea,axis=1)
        
        # Start polynomializing
        poly = PolynomialFeatures(degree=k,include_bias=False)
        train_num_1 = poly.fit_transform(train_num_1)
        test_num_1 = poly.fit_transform(test_num_1)
        
        train = np.column_stack((train_num_1,train_num_0,train_cat))
        del train_num_1,train_num_0,train_cat
        test = np.column_stack((test_num_1,test_num_0,test_cat))
        del test_num_1,test_num_0,test_cat
        if to_csv==True:
            pd.DataFrame(train).to_csv('X_train_poly'+str(k)+'_var'+str(v)+'.csv',index=False)
            pd.DataFrame(test).to_csv('X_test_poly'+str(k)+'_var'+str(v)+'.csv',index=False)
        return train, test, train_target, test_id

#train=pd.read_csv('train.csv')
#test=pd.read_csv('test.csv')
#a,b,c,d=preprocess(train.iloc[0:5,:],test.iloc[0:5,:],k=2,v=7,poly=False,to_csv=False)
#%%
''' #not used part        
k=3; v=9
def preprocess_train(train,k=3,v=9,polynome=True):    
    """
    Inputs:
    train must be obtained by pd.read_csv
    We will polynomialize to degree k the v most important numerical variables, if polynome = True
    Example: k = 2, v = 10, polynome=True, note: we need v<=26 (or v<26, I'm not sure)
    If polynome=False, then k and v can be anything, they have no effect.

    Outputs:
    List of 4 elements:
    - train_es: the processed training matrix, in format numpy, without id and target
    - train_target: type numpy
    """
    # Read data, back up necessary but cubersome variables (id,target) then throw them away
    train_id = train['id'].values
    train_target = train['target'].values
   
    train = train.drop('target',axis=1)
    train = train.drop('id',axis=1)

    # Separating num and cat in train and test
    heap = train
    del train
    cat_columns = [col for col in heap.columns if (('bin' in col)|('cat' in col))]
    num_columns = [col for col in heap.columns if (('bin' not in col) and ('cat' not in col) and (col!='id') and (col!='target'))]
    heap_num = heap[num_columns].values
    heap_cat = heap[cat_columns]
    heap_cat = pd.get_dummies(heap_cat,columns=heap_cat.columns,drop_first=True).values
    train_num = heap_num
    train_cat = heap_cat
    del heap_cat, heap_num
    
    # Imputing missing numerical values by their means
    imp = Imputer(missing_values=-1,strategy='mean',axis=0)
    imp.fit(train_num)
    train_num = imp.transform(train_num)

    # Scaling data, including dummy columns
    mean_train_num = np.mean(train_num,axis=0)
    sd_train_num = np.std(train_num,axis=0)
    train_num = (train_num - mean_train_num)/sd_train_num
    
    
    #mean_train_cat = np.mean(train_cat,axis=0)
    #sd_train_cat = np.std(train_cat,axis=0)
    #train_cat = (train_cat - mean_train_cat)/sd_train_cat
    #test_cat = (test_cat - mean_train_cat)/sd_train_cat

    if (polynome):
        # Choosing v numerical variables most correlated with the target in the training set, then polynomialize them to order k
        train_target_scl = (train_target-np.mean(train_target))/np.std(train_target)
        fea_cov = np.abs(train_target_scl.dot(train_num)) #fea_cov[i] is abs of the not-divided-by-n covariance between the scaled target
        # and the feature [i] of the training set
        imp_fea = np.argsort(-fea_cov)[0:v] #the v most important features
        heap_num =train_num
        heap_num_1 = heap_num[:,imp_fea] # we will polynomialize this matrix
        heap_num_0 = np.delete(heap_num,imp_fea,axis=1)
        # Start polynomializing
        poly = PolynomialFeatures(degree=k,include_bias=False)
        heap_num_1 = poly.fit_transform(heap_num_1)
        # Reseparate
        heap_num = np.concatenate((heap_num_0,heap_num_1),axis=1)
        train_num = heap_num
        del heap_num
        del heap_num_0, heap_num_1, heap
       

    # Give out the processed matrices
    train_es = np.concatenate((train_num,train_cat),axis=1) #esential of the processed training matrix, without id and target
    del train_num,train_cat
    
    return train_es,train_target

X_train, X_test, y_train, y_test = train_test_split(train_es, train_target, test_size=0.18, random_state=None)


# 0.2661 cho bac 2, 254 cho bac 3 voi 9 bien
lo =LogisticRegression()
lo.fit(X_train,y_train)
lr = lo.predict_proba(X_test)[:,1]
gini(y_test,lr)

#ada
from Ada_boost import Ada_test
#bac 3 voi 9 bien
#ada 3 too slow
ada3 = Ada_test1(X_train, y_train, X_test, y_test,D = 3,T = 500,balance=0)
ada7 = Ada_test1(X_train, y_train, X_test, y_test,D = 7,T = 500,balance=0)
'''