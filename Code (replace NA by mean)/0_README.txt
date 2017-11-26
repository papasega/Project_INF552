The file main.py implements the logistic regression and exports the output csv file, with the following parameters:

- C: the inverse penalty parameter (view http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html). The lower C is, the less likely there is overfitting, but the more likely there is underfitting.
- v: the number of variables to be polynomialized (choosen according to importance)
- k: the polynomial degree at to which these variables are polynomialized

The file cv_values.txt contains the list of parameters (C,v,k) used for testing. The testing program is implemented in optimize_cv.py (using cross-validation method).

THe file Gini_result.xlsx gives the result of my tests.