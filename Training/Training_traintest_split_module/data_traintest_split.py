#!/usr/bin/env python
# coding: utf-8

# In[1]:


from Training.Training_preprocessing_module.preprocessing_training import preprocessing_training_class
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Training.Training_log_module.training_log import train_logging
from sklearn.preprocessing import StandardScaler
import pickle

class data_splitting:
    def __init__(self):
        self.pre_obj = preprocessing_training_class()
        

        self.log_filename = 'training_log.txt'
        self.log_obj = train_logging(self.log_filename)
        
        self.log_obj.true_log('Train_Test_Split Operation starts from here ******')

    def traintest_split(self):
        dataframe=self.pre_obj.outliers_removal()
        
        self.log_obj.true_log('Reading standardized data into dataframe')
        
        try:
            X = dataframe.drop(columns='Concrete_compressive _strength')
            Y = dataframe['Concrete_compressive _strength']
            
            self.log_obj.true_log('Separating Independent and dependent features into separate variables X and Y')
            
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 101)
            
            self.log_obj.true_log('Performing train_test_splt with 25% test_data size and 75% train_data size')
        
            return x_train, x_test, y_train, y_test
        
        except Exception as e:
            self.log_obj.error_log(f'Unable to perform train test split due to this error: {str(e)}')
            
    
    def standardization(self):
        x_train, x_test, y_train, y_test=self.traintest_split()
        
        std_obj = StandardScaler()

        x_std_train = std_obj.fit_transform(x_train)
        x_std_test = std_obj.fit_transform(x_test)
                
        with open('standard_scaler.pkl','wb') as std_file:
            pickle.dump(std_obj,std_file)
        return x_std_train,x_std_test,y_train,y_test


# In[ ]:




