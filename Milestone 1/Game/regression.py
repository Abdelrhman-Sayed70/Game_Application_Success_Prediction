from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error
from preprocessing import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


saved_RF = None
saved_DTreg = None
saved_GBR = None
saved_features = None

def plots_func(act, pred):
    plt.scatter(act, pred, c='blue')
    p1 = max(max(pred), max(act))
    p2 = min(min(pred), min(act))
    plt.plot([p1, p2], [p1, p2], 'black')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()

# STEP 1: Read data
df = pd.read_csv('games-regression-dataset.csv')

# STEP 2: Drop null rows and duplicates
DropNullRows(df)
DuplicatesDetectionAndRemoval(df)

# STEP 3: Split features and target
X = df.drop(columns='Average User Rating', inplace=False)
Y = df['Average User Rating']

# STEP 4: S into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)

# STEP 5: Preprocess train dataset
X_train = base(X_train)
X_train = on_train(X_train)
selected_f = feature_selection_regression(X_train, Y_train)
X_train = X_train[selected_f]

X_test = base(X_test)
X_test = on_test(X_test)
X_test = X_test[selected_f]

X = base(X)
X = on_train(X)

# STEP 6: Train Models
rf = RandomForestRegressor()
rf.fit(X_train, Y_train)
p = rf.predict(X_train)
print("Random Forest Model accuracy Train ", r2_score(Y_train, p))
print('Random Forest Model MSE Train ', metrics.mean_squared_error(Y_train, p))
plots_func(Y_train, p)

# Grid Search for DecisionTreeRegressor
param_grid = {
    'max_depth': [2, 4, 6, 8],
    'min_samples_split': [2, 4, 6, 8],
    'min_samples_leaf': [1, 2, 3, 4],
}
grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, Y_train)
best_params = grid_search.best_params_
DTregressor = DecisionTreeRegressor(**best_params, random_state=42)
DTregressor.fit(X_train, Y_train)
p = DTregressor.predict(X_train)
print("Decision Tree Regressor Model accuracy Train ", r2_score(Y_train, p))
print('Decision Tree Regressor Model MSE Train ', metrics.mean_squared_error(Y_train, p))
plots_func(Y_train, p)

GBR = GradientBoostingRegressor(random_state=42)
GBR.fit(X_train, Y_train)
p = GBR.predict(X_train)
print("Gradient Boosting Regressor Model accuracy Train ", r2_score(Y_train, p))
print('Gradient Boosting Regressor Model MSE Train ', metrics.mean_squared_error(Y_train, p))
plots_func(Y_train, p)

# STEP 7: Preprocess test dataset


# STEP 8: Evaluate Models
p = rf.predict(X_test)
print("Random Forest Model accuracy Test", r2_score(Y_test, p))
print('Random Forest Model MSE Test', metrics.mean_squared_error(Y_test, p))
plots_func(Y_test, p)

p = DTregressor.predict(X_test)
print("Decision Tree Regressor Model accuracy Test ", r2_score(Y_test, p))
print('Decision Tree Regressor Model MSE Test ', metrics.mean_squared_error(Y_test, p))
plots_func(Y_test, p)

p = GBR.predict(X_test)
print("Gradient Boosting Regressor Model accuracy Test ", r2_score(Y_test, p))
print('Gradient Boosting Regressor Model MSE Test ', metrics.mean_squared_error(Y_test, p))
plots_func(Y_test, p)

# STEP 9: Preprocess the whole dataset

selected_f = feature_selection_regression(X, Y)
X = X[selected_f]

# STEP 10: Train Models with the whole dataset
rf.fit(X, Y)
p = rf.predict(X)
print("Random Forest Model accuracy overall ", r2_score(Y, p))
print('Random Forest Model MSE overall ', metrics.mean_squared_error(Y, p))

DTregressor.fit(X, Y)
p = DTregressor.predict(X)
print("Decision Tree Regressor Model accuracy overall", r2_score(Y, p))
print('Decision Tree Regressor Model accuracy overall', metrics.mean_squared_error(Y, p))

GBR.fit(X, Y)
p = GBR.predict(X)
print("Gradient Boosting Regressor Model accuracy overall ", r2_score(Y, p))
print('Gradient Boosting Regressor Model MSE overall ', metrics.mean_squared_error(Y, p))

# STEP 11: Save the models
saved_RF = pickle.dumps(rf)
saved_DTreg = pickle.dumps(DTregressor)
saved_GBR = pickle.dumps(GBR)
saved_features = pickle.dumps(selected_f)

