# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:33:06 2021

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
np.mean(cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error', cv= 3)) # NMAE = -60.36060366663539

# Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(min_samples_leaf=.01)
dtr.fit(X_train,y_train)
np.mean(cross_val_score(dtr,X_train,y_train, scoring = 'neg_mean_absolute_error', cv = 3)) # NMAE = -58.64131895872581

# Optimize Random Forest, Decision Tree Regressor
# GridsearchCV & Decision Tree Regressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

depths = np.arange(1, 10)
num_leafs = [1, 5, 10, 15, 20]
pipe_tree = make_pipeline(DecisionTreeRegressor())
parameters = [{'decisiontreeregressor__max_depth':depths,
              'decisiontreeregressor__min_samples_leaf':num_leafs}]
gs = GridSearchCV(pipe_tree, parameters, scoring = 'neg_mean_absolute_error',cv = 3)
gs.fit(X_train,y_train)

gs.best_score_ # -57.95886783391725
gs.best_estimator_
# Pipeline(memory=None,
#          steps=[('decisiontreeregressor',
#                  DecisionTreeRegressor(criterion='mse', max_depth=8,
#                                        max_features=None, max_leaf_nodes=None,
#                                        min_impurity_decrease=0.0,
#                                        min_impurity_split=None,
#                                        min_samples_leaf=10, min_samples_split=2,
#                                        min_weight_fraction_leaf=0.0,
#                                        presort=False, random_state=None,
#                                        splitter='best'))], verbose=False)

# GridsearchCV & Random Forest
pipe_tree1 = make_pipeline(RandomForestRegressor())
num_estimators = np.arange(10, 30)
max_features = ['auto', 'log2', 'sqrt']

rf_parameters = [{
                 'randomforestregressor__n_estimators': num_estimators,
                 'randomforestregressor__max_features' : max_features
                }]

gs = GridSearchCV(pipe_tree1, rf_parameters, scoring = 'neg_mean_absolute_error', cv = 3)
gs.fit(X_train,y_train)

gs.best_score_ # -56.61622630874629
gs.best_estimator_
# Pipeline(memory=None,
#          steps=[('randomforestregressor',
#                  RandomForestRegressor(bootstrap=True, criterion='mse',
#                                        max_depth=None, max_features='log2',
#                                        max_leaf_nodes=None,
#                                        min_impurity_decrease=0.0,
#                                        min_impurity_split=None,
#                                        min_samples_leaf=1, min_samples_split=2,
#                                        min_weight_fraction_leaf=0.0,
#                                        n_estimators=29, n_jobs=None,
#                                        oob_score=False, random_state=None,
#                                        verbose=0, warm_start=False))], verbose=False)

# Predict
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = rf.predict(X_test)
tpred_dtr = dtr.predict(X_test)
tpred_dtr_gs = gs.best_estimator_.predict(X_test)
tpred_rf_gs = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,tpred_lm) # Linear Regression, MAE = 66.33930418823329
mean_absolute_error(y_test,tpred_lml) # Lasso Regression, MAE = 66.332949607733
mean_absolute_error(y_test,tpred_rf) # Random Forest, MAE = 60.94088105473147
mean_absolute_error(y_test,tpred_rf_gs) # GridsearchCV &Random Forest, MAE = 57.38808094257923
mean_absolute_error(y_test,tpred_dtr) # Decision Tree Regressor, MAE = 59.959636525602555
mean_absolute_error(y_test,tpred_dtr_gs) # GridsearchCV & Decision Tree Regressor, MAE = 58.57245099686004
