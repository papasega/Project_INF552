import pandas as pd
from gini import gini

data = pd.read_csv('submission_.csv')
X = data.iloc[:,1]
print('\ngini coefficient := {}'.format(gini(X.values)))