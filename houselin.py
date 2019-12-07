# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 23:12:06 2019

@author: dell
"""
import pandas as pd
df = pd.read_csv('abc.csv')

df1 =df.drop(df.columns[0],axis=1)

df1.drop(df1.iloc[:, 10:15], inplace = True, axis = 1)

################# ML part #################################
#### split_train_test#################

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df1, test_size=0.2, random_state=42)

X=train_set.drop('median_house_value',axis=1)
y=train_set['median_house_value']

X_test=test_set.drop('median_house_value',axis=1)
y_test=test_set['median_house_value']

################################ Linear-Regression########################################
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
lin_reg.score(X,y)
#########RMSE Train Data############
from sklearn.metrics import mean_squared_error
import numpy as np
y_pred_train = lin_reg.predict(X)
lin_mse_train = mean_squared_error(y, y_pred_train)
lin_rmse_train = np.sqrt(lin_mse_train)
lin_rmse_train
from matplotlib import pyplot as plt
plt.scatter(y,y_pred_train)
plt.xlabel(“True Values”)
plt.ylabel(“Predictions”)
########################### RMSE test data set #########
y_pred_test = lin_reg.predict(X_test)
lin_reg.fit(X_test,y_test)
lin_reg.score(X_test,y_test)

lin_mse_test = mean_squared_error(y_test, y_pred_test)
lin_rmse_test = np.sqrt(lin_mse_test)
lin_rmse_test


##########Ridge Regression##########
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.score(X, y)


y_ridge_train = ridge_reg.predict(X)
ridge_mse_train = mean_squared_error(y, y_ridge_train)
ridge_rmse_train = np.sqrt(ridge_mse_train)
ridge_rmse_train
from matplotlib import pyplot as plt
plt.scatter(y,y_ridge_train)
####### RMSE test data set #
y_pred_test = ridge_reg.predict(X_test)
ridge_reg.fit(X_test,y_test)
ridge_reg.score(X_test,y_test)
ridge_mse_test = mean_squared_error(y_test, y_pred_test)
ridge_rmse_test = np.sqrt(ridge_mse_test)
ridge_rmse_test

########lasso Regression######

from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.score(X, y)

y_lasso_train = lasso_reg.predict(X)
lasso_mse_train = mean_squared_error(y, y_lasso_train)
lasso_rmse_train = np.sqrt(lasso_mse_train)
lasso_rmse_train
plt.scatter(y,y_lasso_train)
###RMSE test data set #
y_lasso_test = lasso_reg.predict(X_test)
lasso_reg.fit(X_test,y_test)
lasso_reg.score(X_test,y_test)
lasso_mse_test = mean_squared_error(y_test, y_lasso_test)
lasso_rmse_test = np.sqrt(lasso_mse_test)
lasso_rmse_test


#########ElasticNet#######33


from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.score(X, y)

y_elastic_net_train = elastic_net.predict(X)
elastic_net_mse_train = mean_squared_error(y, y_elastic_net_train)
elastic_net_rmse_train = np.sqrt(elastic_net_mse_train)
elastic_net_rmse_train
plt.scatter(y,y_elastic_net_train)
###RMSE test data set #
y_elastic_net_test = elastic_net.predict(X_test)
elastic_net.fit(X_test,y_test)
elastic_net.score(X_test,y_test)
elastic_net_mse_test = mean_squared_error(y_test, y_elastic_net_test)
elastic_net_rmse_test = np.sqrt(elastic_net_mse_test)
elastic_net_rmse_test

##############SGDRegressor#############

from sklearn.linear_model import SGDRegressor
SGD_reg = SGDRegressor(max_iter=1000, tol=1e-3)
SGD_reg.fit(X, y)
SGD_reg.score(X, y)

y_pred_SGD_train = SGD_reg.predict(X)
SGD_mse_train = mean_squared_error(y, y_pred_SGD_train)
SGD_rmse_train = np.sqrt(SGD_mse_train)
SGD_rmse_train
plt.scatter(y,y_pred_SGD_train )

train_rmse=[lin_rmse_train,ridge_rmse_train,lasso_rmse_train,elastic_net_rmse_train,SGD_rmse_train]
aa=pd.DataFrame(train_rmse)
###RMSE test data set #
y_pred_SGD_test = SGD_reg.predict(X_test)
SGD_reg.fit(X_test,y_test)
SGD_reg.score(X_test,y_test)
SGD_mse_test = mean_squared_error(y_test, y_pred_SGD_test)
SGD_rmse_test = np.sqrt(elastic_net_mse_test)
SGD_rmse_test

################################# CSV File ############################################
model=['Linear','Ridge','Lasso','Elastic_Net','SGD']
aa4=pd.DataFrame(model)
Result=pd.concat([aa4,aa,aa1,aa2,aa3],axis=1)
Result.columns=['MODEL','RMSE_TRAIN','RMSE_TEST','R^2_SCORE_TRAIN','R^2_SCORE_TEST']
Result.to_csv("Results.csv",index=False)


##########cross validation#######
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
######Single metric evaluation using cross_validate
cv_results = cross_validate(lasso, X, y, cv=3)
sorted(cv_results.keys())                         
cv_results['test_score']    

####3Multiple metric evaluation using cross_validate (please refer the scoring parameter doc for more information)

scores = cross_validate(lasso, X, y, cv=3,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)
print(scores['test_neg_mean_squared_error'])      

print(scores['train_r2'])                         

