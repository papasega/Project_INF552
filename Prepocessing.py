#%%
import numpy as np
import pandas as pd
import pylab as plt

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import PolynomialFeatures

### Fetch the data and load it in pandas
data = pd.read_csv('train.csv')
#test
test = pd.read_csv('test.csv')



def preprocessing(data,test, to_csv=False):

    #train
    X = data.drop('target',axis=1)
    X = X.drop('id',axis=1)
    X_train = X.values
        
    y = data['target'].values
    y_train=2*y-1
        
        
    #test
    ids= test['id']
    test = test.drop('id',axis=1)
    X_test = test.values
    
    #separate cat with pandas
    non_cats=[]
    cats=[]
    
    
    for i in range(len(X.columns.values)):
        if 'cat' in X.columns.values[i]:
            cats.append(i)
        else:
            non_cats.append(i)
    
    X_cat=X.iloc[:,cats].values
    X_non_cat=X.iloc[:,non_cats].values
    
    test_cat=test.iloc[:,cats].values
    test_non_cat=test.iloc[:,non_cats].values
    
    
    #one hot encoder 
    enc = OneHotEncoder()
    enc.fit(X_cat+1)
    
    X_cat_enc=enc.transform(X_cat+1).toarray()
    test_cat_enc=enc.transform(test_cat+1).toarray()

    '''enc, no touch to non cat'''
    train_enc=pd.DataFrame(np.column_stack((X_non_cat,X_cat_enc)))
    test_enc=pd.DataFrame(np.column_stack((test_non_cat,test_cat_enc)))


    '''cat enc, non cat to mean'''
    
    imp = Imputer(missing_values=-1,strategy='mean',axis=0)
    imp.fit(X_non_cat)

    X_non_cat_mean=imp.transform(X_non_cat)
    test_non_cat_mean=imp.transform(test_non_cat)
    
    X_train_m=np.column_stack((X_non_cat_mean,X_cat_enc))
    X_test_m=np.column_stack((test_non_cat_mean,test_cat_enc))
    
    train_enc_mean=pd.DataFrame(X_train_m)
    test_enc_mean=pd.DataFrame(X_test_m)
    
    #print enc
    if to_csv == True:
        train_enc.to_csv('train_enc.csv',index=False)
        test_enc.to_csv('test_enc.csv',index=False)
        pd.DataFrame(data=ids,columns=['id']).to_csv('id.csv',index=False)
        pd.DataFrame(y).to_csv('y_train.csv',index=False)
        train_enc_mean.to_csv('train_enc_mean.csv',index=False)
        test_enc_mean.to_csv('test_enc_mean.csv',index=False)

    return X_train_m, y, X_test_m, ids, X_non_cat_mean, X_cat_enc, test_non_cat_mean, test_cat_enc 


'''poly big memory'''
    
def poly(X_non_cat_mean, test_non_cat_mean, degree=2)    :
    #fit
    poly = PolynomialFeatures(degree)
    poly.fit(X_non_cat_mean)
    #poly
    X_non_cat_mean_poly = poly.transform(X_non_cat_mean_poly)
    test_non_cat_mean = poly.transform(test_non_cat_mean)
    
    return X_non_cat_mean_poly, test_non_cat_mean_poly
    
    



