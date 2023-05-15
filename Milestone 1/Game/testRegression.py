from sklearn import metrics
from sklearn.metrics import r2_score
import pandas as pd
rf_test = None
DT_test = None
GBR_test = None
f = None
from preprocessing import *



ms1_df = pd.read_csv('test_regression_data.csv')
X = ms1_df.drop(columns='Average User Rating', inplace=False)
Y = ms1_df['Average User Rating']
X = base(X)
X = on_test(X)
f = load("selected_f.pkl")

X = X[f]

rf_test = load("rf.pkl")
p = rf_test.predict(X)
print("Random Forest Model accuracy Test File ", r2_score(Y, p))
print('Random Forest Model MSE Test File ', metrics.mean_squared_error(Y, p))


DT_test = load("DTregressor.pkl")
p = DT_test.predict(X)
print("Decision Tree Regressor Model accuracy Test File ", r2_score(Y, p))
print('Decision Tree Regressor Model MSE Test File ', metrics.mean_squared_error(Y, p))


GBR_test = load("GBR.pkl")
p = GBR_test.predict(X)
print("Gradient Boosting Regressor Model accuracy Test File ", r2_score(Y, p))
print('Gradient Boosting Regressor Model MSE Test File ', metrics.mean_squared_error(Y, p))