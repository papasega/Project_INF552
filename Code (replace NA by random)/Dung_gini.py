

from sklearn.metrics import roc_auc_score

def gini(true,predict):
    return 2*roc_auc_score(true,predict)-1

#a=np.array([1,0,1])
#=np.array([1,2,1])
#roc_auc_score(a,a)
#gini(np.array([1,0,1,0,0]),2*np.array([-1.5,-1.5,2.5,3,9])+1)















