import pandas as pd

data = pd.read_csv('submission.csv')
X = data.iloc[:,1]
u = 1 - X.values
ids = data.iloc[:,0]
df = pd.DataFrame({"id": ids, "target": u})
df.to_csv("../TENSORFLOW/submission_.csv", index=False)