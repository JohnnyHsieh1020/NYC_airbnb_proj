# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 14:59:57 2021

@author: Johnny Hsieh
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.read_csv('dataset/eda_data.csv')
df = df.drop(columns=['Unnamed: 0'], axis = 1)

# Prepare data 
encode = preprocessing.LabelEncoder()
encode.fit(df.neighbourhood_group)
df.neighbourhood_group=encode.transform(df.neighbourhood_group)

encode = preprocessing.LabelEncoder()
encode.fit(df.neighbourhood)
df.neighbourhood=encode.transform(df.neighbourhood)

encode = preprocessing.LabelEncoder()
encode.fit(df.room_type)
df.room_type=encode.transform(df.room_type)

# Split data
from sklearn.model_selection import train_test_split

X = df.drop(columns=['price'], axis = 1)
y = df.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Model Building
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

# Linear Regression
lm = LinearRegression()
lm.fit(X_train, y_train)
np.mean(cross_val_score(lm,X_train,y_train, scoring = 'neg_mean_absolute_error', cv = 3)) # NMAE = -64.80233916737052

# Lasso Regression
lm_l = Lasso(alpha=0.7)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv = 3))

# 找出使 NMAE 最大的 alpha
# a = []
# e = []

# for i in range(1, 20):
#   a.append(i/10)    
#   lm_l = Lasso(alpha = (i/10))
#   e.append(np.mean(cross_val_score(lm_l,X_train,y_train, scoring = 'neg_mean_absolute_error', cv = 3)))

# err = tuple(zip(a,e))
# df_err = pd.DataFrame(err, columns = ['alpha','error'])
# df_err[df_err.error == max(df_err.error)] # alpha = 0.7, NMAE = -64.78917484002328

# Random Forest 
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()
rf.fit(X_train,y_train)
np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3)) # NMAE = -60.443919303573665

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(min_samples_leaf=.01)
dtr.fit(X_train,y_train)
np.mean(cross_val_score(dtr,X_train,y_train, scoring = 'neg_mean_absolute_error', cv = 3)) # NMAE = -58.64131895872581

# GridsearchCV & Random Forest
from sklearn.model_selection import GridSearchCV

num_estimators = np.arange(1,10)
depths = np.arange(1,10)
num_leafs = np.arange(1,10)
max_features = ['auto', 'log2', 'sqrt']
rf_parameters = [{
                 'n_estimators': num_estimators,
                 'max_depth' : depths,
                 'min_samples_leaf' : num_leafs,
                 'max_features' : max_features}]

rf_gs = GridSearchCV(RandomForestRegressor(), rf_parameters, scoring = 'neg_mean_absolute_error', cv = 3)
rf_gs.fit(X_train,y_train)

rf_gs.best_score_ # -56.048108459722435
rf_gs.best_estimator_
# RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=9,
#                       max_features='auto', max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=9, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, n_estimators=9, n_jobs=None,
#                       oob_score=False, random_state=None, verbose=0,
#                       warm_start=False)

# GridsearchCV & Decision Tree Regressor
depths = np.arange(1,10)
num_leafs = np.arange(1,10)
dt_parameters = [{'max_depth' : depths,
                  'min_samples_leaf' : num_leafs}]

dt_gs = GridSearchCV(DecisionTreeRegressor(), dt_parameters, scoring = 'neg_mean_absolute_error',cv = 3)
dt_gs.fit(X_train,y_train)

dt_gs.best_score_ # -57.86494271844868
dt_gs.best_estimator_
# DecisionTreeRegressor(criterion='mse', max_depth=8, max_features=None,
#                       max_leaf_nodes=None, min_impurity_decrease=0.0,
#                       min_impurity_split=None, min_samples_leaf=9,
#                       min_samples_split=2, min_weight_fraction_leaf=0.0,
#                       presort=False, random_state=None, splitter='best')


# Predict
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = rf.predict(X_test)
tpred_rf_gs = rf_gs.best_estimator_.predict(X_test)
tpred_dtr = dtr.predict(X_test)
tpred_dtr_gs = dt_gs.best_estimator_.predict(X_test)

# Performance
from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,tpred_lm) # Linear Regression, MAE = 66.33930418823329
mean_absolute_error(y_test,tpred_lml) # Lasso Regression, MAE = 66.332949607733
mean_absolute_error(y_test,tpred_rf) # Random Forest, MAE = 60.953719964861605
mean_absolute_error(y_test,tpred_rf_gs) # GridsearchCV & Random Forest, MAE = 56.940950944644406
mean_absolute_error(y_test,tpred_dtr) # Decision Tree Regressor, MAE = 59.959636525602555
mean_absolute_error(y_test,tpred_dtr_gs) # GridsearchCV & Decision Tree Regressor, MAE = 58.603089023498626