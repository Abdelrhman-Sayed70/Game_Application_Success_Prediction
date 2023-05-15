from preprocessing import *
import pandas as pd


rbf = None
svc = None
lin = None
poly = None
f = None

ms2_df = pd.read_csv('test_classification2.csv')
X = ms2_df.drop(columns='Rate', inplace=False)
Y = ms2_df['Rate']
X = base(X)
X = on_test(X)
f = load("selected_f_classification.pkl")
X = X[f]



logistic = load("logistic_regression.pkl")
Ypred = logistic.predict(X)
accuracy = np.mean(Ypred == Y)
print("logistic_regression Test File: ", accuracy)

DT = load("DT_classifer.pkl")
Ypred = DT.predict(X)
accuracy = np.mean(Ypred == Y)
print("DecisionTreeClassifier Test File: ", accuracy)

rbf = load("rbf_svc.pkl")
Ypred = rbf.predict(X)
accuracy = np.mean(Ypred == Y)
print("SVC with RBF kernel Test File: ", accuracy)


svc = load("svc.pkl")
Ypred = svc.predict(X)
accuracy = np.mean(Ypred == Y)
print("SVC with linear kernel Test File: ", accuracy)


lin = load("lin_svc.pkl")
Ypred = lin.predict(X)
accuracy = np.mean(Ypred == Y)
print("LinearSVC (linear kernel) Test File: ", accuracy)


poly = load("poly_svc.pkl")
Ypred = poly.predict(X)
accuracy = np.mean(Ypred == Y)
print("SVC with polynomial (degree 3) kernel Test File: ", accuracy)

