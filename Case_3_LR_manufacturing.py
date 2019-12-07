# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 05:38:35 2019

@author: NETHRA
"""

import pandas as pd
import numpy as np
df=pd.read_excel('Case_3_LR_manufacturing.xls')
df.head()
df.describe()
df1=df.describe()
df1.to_excel('case3.xlsx')
df1.to_csv("describe.csv")
df1.to_excel('abc.xlsx')
df.columns=['cement','slag','Fly_Ash','water','super','coarse','fine','age','concrete']
#df2
df2=df.groupby('concrete').size() 
df2
df.corr()
df.skew()
df.dtypes
df.isnull().sum()    #NA values in data
df = df.replace(np.nan, 0)
df['concrete'].mean() ###35.81783582611362
df['concrete'].mode()
df['super'].max()
df['coarse'].max()
df['coarse'].min()
df.hist(column='concrete')
df.hist()
df.hist(column='cement')
df.hist(column='slag')
df.plot.kde()
df.boxplot(column='concrete')
df.plot.density()
df.plot.density(column='concrete')
df.plot()
df.boxplot()
df.boxplot(column='slag')
df.isnull().sum()    #NA values in data
df = df.replace(np.nan, 0)
############out######
#########
#############33model####
df2 =df.drop(df.columns[0],axis=1)

df2.drop(df1.iloc[:, 10:15], inplace = True, axis = 1)
#### split_train_test#################

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df1, test_size=0.2, random_state=42)

X=train_set.iloc[:,0 :-1].values
y=train_set.iloc[:,1]
X=train_set.iloc[:,0 :-1].values
y=train_set.iloc[:,0 :-1].values

X_test=test_set.iloc[:,0 :-1].values
y_test=test_set.iloc[:,1].values
################################ Linear-Regression########################################
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
lin_reg.score(X,y)
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



