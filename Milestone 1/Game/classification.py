import warnings
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import time
from preprocessing import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
#ignor warning
warnings.filterwarnings(action='ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')
saved_rbf = None
saved_svc = None
saved_lin = None
saved_poly = None
saved_features = None
accuracyvalues=[]
accuracylabels=[]
timetest=[]
timetestlabels=[]
timetrain=[]
timetrainlabels=[]
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
print("              train accuracy              ")
start_time = time.time()
logistic_regression =LogisticRegression(max_iter=1000).fit(X_train, Y_train)
end_time = time.time()
training_time = end_time - start_time
timetrain.append(training_time)
timetrainlabels.append("logistic_regression  ")
Ypred = logistic_regression.predict(X_train)
accuracy = np.mean(Ypred == Y_train)
print("logistic_regression Train accuracy:", accuracy)



start_time = time.time()
DT_classifer =DecisionTreeClassifier(max_depth=20).fit(X_train, Y_train)
end_time = time.time()
training_time = end_time - start_time
timetrain.append(training_time)
timetrainlabels.append("DecisionTreeClassifier")
Ypred = DT_classifer.predict(X_train)
accuracy = np.mean(Ypred == Y_train)
print("DecisionTreeClassifier Train accuracy:", accuracy)

C = 1  # SVM regularization parameter

start_time = time.time()
rbf_svc = svm.SVC(kernel='rbf', gamma=0.8, C=C).fit(X_train, Y_train)
end_time = time.time()
training_time = end_time - start_time
Ypred = rbf_svc.predict(X_train)
timetrain.append(training_time)
timetrainlabels.append("rbf_svm")
accuracy = np.mean(Ypred == Y_train)
print("SVC with RBF kernel Train ", accuracy)

start_time = time.time()
svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
end_time = time.time()
training_time = end_time - start_time
timetrain.append(training_time)
timetrainlabels.append("linesr_svm")
Ypred = svc.predict(X_train)
accuracy = np.mean(Ypred == Y_train)
print("SVC with linear kernel Train ", accuracy)

start_time = time.time()
lin_svc = svm.LinearSVC(C=C).fit(X_train, Y_train)
end_time = time.time()
training_time = end_time - start_time
timetrain.append(training_time)
timetrainlabels.append("lin_svc")
Ypred = lin_svc.predict(X_train)
accuracy = np.mean(Ypred == Y_train)
print("LinearSVC (linear kernel) Train ", accuracy)

start_time = time.time()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, Y_train)
end_time = time.time()
training_time = end_time - start_time
timetrain.append(training_time)
timetrainlabels.append("poly_svc")
Ypred = poly_svc.predict(X_train)
accuracy = np.mean(Ypred == Y_train)
print("SVC with polynomial (degree 3) kernel Train ", accuracy)


# STEP 7: Preprocess test dataset

X_test = X_test[selected_f]

# STEP 8: Evaluate Models
print("\n              test accuracy              ")
start_time = time.time()
Ypred = logistic_regression.predict(X_test)
end_time = time.time()
test_time = end_time - start_time
timetest.append(test_time)
timetestlabels.append("logistic_regression  ")
accuracy = np.mean(Ypred == Y_test)
accuracyvalues.append(accuracy)
accuracylabels.append("logistic-regression")
print("logistic_regression Test accuracy:", accuracy)


start_time = time.time()
Ypred = DT_classifer.predict(X_test)
end_time = time.time()
test_time = end_time - start_time
timetest.append(test_time)
timetestlabels.append("DecisionTreeClassifier  ")
accuracy = np.mean(Ypred == Y_test)
accuracyvalues.append(accuracy)
accuracylabels.append("DecisionTreeClassifier")
print("DecisionTreeClassifier Test accuracy:", accuracy)

start_time = time.time()
Ypred = rbf_svc.predict(X_test)
end_time = time.time()
training_time = end_time - start_time
timetest.append(training_time)
timetestlabels.append("rbf_svm")
accuracy = np.mean(Ypred == Y_test)
accuracyvalues.append(accuracy)
accuracylabels.append("rbf_svc")
print("SVC with RBF kernel Test ", accuracy)

start_time = time.time()
Ypred = svc.predict(X_test)
end_time = time.time()
accuracy = np.mean(Ypred == Y_test)
training_time = end_time - start_time
timetest.append(training_time)
timetestlabels.append("linear_svm")
accuracyvalues.append(accuracy)
accuracylabels.append("svc")
print("SVC with linear kernel Test ", accuracy)

start_time = time.time()
Ypred = lin_svc.predict(X_test)
end_time = time.time()
accuracy = np.mean(Ypred == Y_test)
training_time = end_time - start_time
timetest.append(training_time)
timetestlabels.append("linear_svc")
accuracyvalues.append(accuracy)
accuracylabels.append("LinearSVC")
print("LinearSVC (linear kernel) Test ", accuracy)

start_time = time.time()
Ypred = poly_svc.predict(X_test)
end_time = time.time()
accuracy = np.mean(Ypred == Y_test)
test_time = end_time - start_time
timetest.append(test_time)
timetestlabels.append("poly_svc")
accuracyvalues.append(accuracy)
accuracylabels.append("poly_csv")
print("SVC with polynomial (degree 3) kernel Test ", accuracy)

# STEP 9: Preprocess the whole dataset
print("\n              whole dataset             ")
selected_f = feature_selection_classification(X, Y)
X = X[selected_f]

# STEP 10: Train Models with the whole dataset
logistic_regression.fit(X, Y)
Ypred =logistic_regression.predict(X)
accuracy = np.mean(Ypred == Y)
print("logistic_regression overall ", accuracy)

DT_classifer.fit(X, Y)
Ypred = DT_classifer.predict(X)
accuracy = np.mean(Ypred == Y)
print("DT_classifer overall ", accuracy)

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
savee(rbf_svc,"rbf_svc.pkl")
savee(svc,     "svc.pkl")
savee(lin_svc,"lin_svc.pkl")
savee(poly_svc,"poly_svc.pkl")
savee(selected_f,"selected_f_classification.pkl")
savee(logistic_regression,"logistic_regression.pkl")
savee(DT_classifer,"DT_classifer.pkl")

#bar plot of accuracy
plt.bar(accuracylabels, accuracyvalues)
plt.title('accuracy of different models')
plt.xlabel('Labels')
plt.ylabel('Values')
plt.show()

#bar plot of train time
plt.bar(timetrainlabels, timetrain)
plt.title('total train time of different models')
plt.xlabel('models')
plt.ylabel('time of train')
plt.show()

#bar plot of train time
plt.bar(timetestlabels, timetest)
plt.title('total test time of different models')
plt.xlabel('models')
plt.ylabel('time of train')
plt.show()