from classification import *

rbf = None
svc = None
lin = None
poly = None
f = None

ms2_df = pd.read_csv('games-classification-dataset.csv')
X = ms2_df.drop(columns='Rate', inplace=False)
Y = ms2_df['Rate']
X = base(X)
X = on_test(X)
f = pickle.loads(saved_features)
X = X[f]

rbf = pickle.loads(saved_rbf)
Ypred = rbf.predict(X)
accuracy = np.mean(Ypred == Y)
print("SVC with RBF kernel Test File ", accuracy)


svc = pickle.loads(saved_svc)
Ypred = svc.predict(X)
accuracy = np.mean(Ypred == Y)
print("SVC with linear kernel Test File ", accuracy)


lin = pickle.loads(saved_lin)
Ypred = lin.predict(X)
accuracy = np.mean(Ypred == Y)
print("LinearSVC (linear kernel) Test File ", accuracy)


poly = pickle.loads(saved_poly)
Ypred = poly.predict(X)
accuracy = np.mean(Ypred == Y)
print("SVC with polynomial (degree 3) kernel Test File ", accuracy)