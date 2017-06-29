import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
import sklearn.metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors

df = pd.read_csv('all_data.csv')
features = ['cough', 'TB', 'tuberculosis', 'sex']
X = df[features]
Y = df['Rate']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=6)

#model = GridSearchCV(LinearRegression(), param_grid = {'fit_intercept': ['True', 'False']})
#model = GridSearchCV(KernelRidge(), cv=5, param_grid = {"alpha": [10, 0.1, 1e-3], "gamma" : np.logspace(-15, 15, 4), 'kernel':['linear', 'rbf']}, scoring='neg_mean_squared_error')
#model = GridSearchCV(Lasso(), cv=5, param_grid = {"alpha": [0.1, 0.3, 0.6, 1.0]})
#model = GridSearchCV(Ridge(), cv=4, param_grid={"alpha": [ 0.1,1,10]}, scoring='neg_mean_squared_error')
#model = GridSearchCV(neighbors.KNeighborsRegressor(), cv=5, param_grid = {'n_neighbors': [5, 15, 30, 45, 60], 'weights' : ['distance', 'uniform']}, scoring = 'neg_mean_squared_error')
#model = GridSearchCV(ElasticNet(), param_grid = {'alpha' : [0.1, 0.3,0.6], 'l1_ratio': [0.1, 0.3, 0.6]}, scoring = 'neg_mean_squared_error')
#model = ElasticNet(alpha=0.1, l1_ratio=0.1)
model = LinearRegression()
model.fit(X_train, Y_train)
#print model.grid_scores_
#print model.best_params_
#print model.sparse_coef_ #for elasticnet
print model.coef_ #for others

Y_pred = model.predict(X_test)

#print various error metrics
print sklearn.metrics.explained_variance_score(Y_test, Y_pred)
print sklearn.metrics.mean_squared_error(Y_test, Y_pred)
print sklearn.metrics.mean_absolute_error(Y_test, Y_pred)
print sklearn.metrics.r2_score(Y_test, Y_pred)
