import pandas as pd
from sklearn import svm
from preprocessing import *
from sklearn.model_selection import train_test_split

saved_rbf = None
saved_svc = None
saved_lin = None
saved_poly = None
saved_features = None

# STEP 1: Read data
df = pd.read_csv('games-classification-dataset.csv')

# STEP 2: Drop null rows and duplicates
DropNullRows(df)
DuplicatesDetectionAndRemoval(df)

# STEP 3: Split features and target
X = df.drop(columns='Rate', inplace=False)
Y = df['Rate']

# STEP 4: S into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, stratify=Y)

# STEP 5: Preprocess train dataset
X_train = base(X_train)
X_train = on_train(X_train)
X_test = base(X_test)
X_test = on_test(X_test)
X = base(X)
X = on_train(X)
selected_f = feature_selection_classification(X_train, Y_train)
X_train = X_train[selected_f]

# STEP 6: Train Models
C = 1  # SVM regularization parameter

rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X_train, Y_train)
Ypred = rbf_svc.predict(X_train)
accuracy = np.mean(Ypred == Y_train)
print("SVC with RBF kernel Train ", accuracy)

svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
Ypred = svc.predict(X_train)
accuracy = np.mean(Ypred == Y_train)
print("SVC with linear kernel Train ", accuracy)

lin_svc = svm.LinearSVC(C=C).fit(X_train, Y_train)
Ypred = lin_svc.predict(X_train)
accuracy = np.mean(Ypred == Y_train)
print("LinearSVC (linear kernel) Train ", accuracy)

poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, Y_train)
Ypred = poly_svc.predict(X_train)
accuracy = np.mean(Ypred == Y_train)
print("SVC with polynomial (degree 3) kernel Train ", accuracy)

# STEP 7: Preprocess test dataset

X_test = X_test[selected_f]

# STEP 8: Evaluate Models
Ypred = rbf_svc.predict(X_test)
accuracy = np.mean(Ypred == Y_test)
print("SVC with RBF kernel Test ", accuracy)

Ypred = svc.predict(X_test)
accuracy = np.mean(Ypred == Y_test)
print("SVC with linear kernel Test ", accuracy)

Ypred = lin_svc.predict(X_test)
accuracy = np.mean(Ypred == Y_test)
print("LinearSVC (linear kernel) Test ", accuracy)

Ypred = poly_svc.predict(X_test)
accuracy = np.mean(Ypred == Y_test)
print("SVC with polynomial (degree 3) kernel Test ", accuracy)

# STEP 9: Preprocess the whole dataset

selected_f = feature_selection_classification(X, Y)
X = X[selected_f]

# STEP 10: Train Models with the whole dataset
rbf_svc.fit(X, Y)
Ypred = rbf_svc.predict(X)
accuracy = np.mean(Ypred == Y)
print("SVC with RBF kernel overall ", accuracy)

svc.fit(X, Y)
Ypred = svc.predict(X)
accuracy = np.mean(Ypred == Y)
print("SVC with linear kernel overall ", accuracy)

lin_svc.fit(X, Y)
Ypred = lin_svc.predict(X)
accuracy = np.mean(Ypred == Y)
print("LinearSVC (linear kernel) overall ", accuracy)

poly_svc.fit(X, Y)
Ypred = poly_svc.predict(X)
accuracy = np.mean(Ypred == Y)
print("SVC with polynomial (degree 3) kernel overall ", accuracy)

# STEP 11: Save the models
saved_rbf = pickle.dumps(rbf_svc)
saved_svc = pickle.dumps(svc)
saved_lin = pickle.dumps(lin_svc)
saved_poly = pickle.dumps(poly_svc)
saved_features = pickle.dumps(selected_f)
