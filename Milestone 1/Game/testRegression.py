from regression import *

rf_test = None
DT_test = None
GBR_test = None
f = None

ms1_df = pd.read_csv('games-regression-dataset.csv')
X = ms1_df.drop(columns='Average User Rating', inplace=False)
Y = ms1_df['Average User Rating']
X = base(X)
X = on_test(X)
f = pickle.loads(saved_features)
X = X[f]

rf_test = pickle.loads(saved_RF)
p = rf_test.predict(X)
print("Random Forest Model accuracy Test File ", r2_score(Y, p))
print('Random Forest Model MSE Test File ', metrics.mean_squared_error(Y, p))


DT_test = pickle.loads(saved_DTreg)
p = DT_test.predict(X)
print("Decision Tree Regressor Model accuracy Test File ", r2_score(Y, p))
print('Decision Tree Regressor Model MSE Test File ', metrics.mean_squared_error(Y, p))


GBR_test = pickle.loads(saved_GBR)
p = GBR_test.predict(X)
print("Gradient Boosting Regressor Model accuracy Test File ", r2_score(Y, p))
print('Gradient Boosting Regressor Model MSE Test File ', metrics.mean_squared_error(Y, p))