#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Training.Training_preprocessing_module.preprocessing_training import preprocessing_training_class
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import pickle
from Training.Training_traintest_split_module.data_traintest_split import data_splitting
import shutil
from Training.Training_log_module.training_log import train_logging
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error

class tuning_training:
    def __init__(self):
        self.split_obj = data_splitting()
        self.x_train, self.x_test, self.y_train, self.y_test = self.split_obj.standardization()

        self.log_filename = 'training_log.txt'
        self.log_obj = train_logging(self.log_filename)
        
        self.log_obj.true_log('Model Tuning and Model Training on various Algorithms operates here ******')
        
    def parameters(self):
        self.rf_parameter = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 100)],
                             'max_features': ['auto', 'sqrt'],
                             'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
                             'min_samples_split': [2, 5, 10],
                             'min_samples_leaf': [1, 2, 4],
                             'bootstrap': [True, False]
                             }
        
        
        return self.rf_parameter
    
    def tune_train(self):
        self.rf_parameter = self.parameters()
        
        rf_reg=RandomForestRegressor()
      
        rf_random = RandomizedSearchCV( estimator = rf_reg, param_distributions = self.rf_parameter, n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs=1)
        rf_random.fit(self.x_train,self.y_train)

        n_estimators_rf = rf_random.best_params_['n_estimators']
        max_features_rf = rf_random.best_params_['max_features']
        max_depth_rf = rf_random.best_params_['max_depth']
        min_sample_split_rf = rf_random.best_params_['min_samples_split']
        min_samples_leaf_rf = rf_random.best_params_['min_samples_leaf']
        bootstrap_rf = rf_random.best_params_['bootstrap']
    
        rf_model=RandomForestRegressor(n_estimators=n_estimators_rf,max_features=max_features_rf,
                                          max_depth=max_depth_rf,min_samples_split=min_sample_split_rf,
                                         min_samples_leaf=min_samples_leaf_rf,bootstrap=bootstrap_rf)
        
        rf_model.fit(self.x_train,self.y_train)
        rf_predict = rf_model.predict(self.x_test)
        score_rf = r2_score(self.y_test,rf_predict)
        
        mse=mean_absolute_error(self.y_test,rf_predict)
        rmse=np.sqrt(mse)
        mae=mean_absolute_error(self.y_test,rf_predict)
        
        with open('randomforest.pkl','wb') as save_file:
            pickle.dump(rf_model,save_file)
        
        return f'r2_score:{round(100*score_rf,3)},RMSE:{round(rmse,3)} and MAE:{round(mae,3)}'

