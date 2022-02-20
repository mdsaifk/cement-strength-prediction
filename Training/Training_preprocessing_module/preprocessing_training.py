#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import shutil
from sklearn.impute import KNNImputer
from Training.Training_log_module.training_log import train_logging
import pickle

class preprocessing_training_class:
    def __init__(self):
        self.file_name = '../html_heroku_deployment/training_data.csv' 
        self.df = pd.read_csv(self.file_name)
        
        self.log_filename = 'training_log.txt'
        self.log_obj = train_logging(self.log_filename)
        
        self.log_obj.true_log('Preprocessing starte from here************')
        
    def check_std_zero(self):
        '''Checking if there is any feature having standard deviation=0
            If yes then we will going to remove that because all the point in that features will be equal to mean and 
            it will cause issue in the calculation
            The dispersion of the data points from the mean will be 0 because actual value equals to mean, 
            it means data is not deviates from mean.
            Values of that column are constant, means whether the strength is good or bad, that values does not changing.
            It means that it has no impact on the target values. So we have to delete that column.
            Hence it is ot necessary for model training as it is not helping in predicting.
            Then we are putting those feature in separate csv file
        '''
        
        self.log_obj.true_log(f'''Defining a fuction which will check if any column has a standard deviation zero or not.
        If it is zero then that column will be removed. The dispersion of the data points from the mean will be 0 because
        actual value equals to mean, it means data is not deviates from mean. Values of that column are constant,
        means whether the strength is good or bad, that values does not changing. It means that it has no impact on the 
        target values. So we have to delete that column.''')
        
        self.df2=self.df.copy()
        self.std_zero = {}
        list_drop_col=[]
        
        self.log_obj.true_log('Checking the column one by one for standard deviation zero')
        for i in list(self.df.columns):
            if self.df2[i].std() == 0: # Chceking for column having zero standard deviation
                self.std_zero[i] = 0 # Filling the diction with key=column name & value=standard deviation
                list_drop_col.append(i)
        
        self.df2.drop(columns = list_drop_col, inplace = True) # Dropping those columns having std zero

        self.log_obj.true_log('Columns having zero standard deviation are dropped.')
        
#         self.std_zero_df = pd.DataFrame([self.std_zero])
#         self.std_zero_df.to_csv('std_zero_data.csv', index = False, header = True)
        
#         self.log_obj.true_log('The columns having zero standard deviation is exported to new csv file')       
        
        return self.df2

        
    def replacing_to_nullvalues(self):
        self.log_obj.true_log('Defining a function that will Replace all the 0 (zero) in every column with null value')
        df = self.check_std_zero()
        null_dic = {}
            
        for feature in list(df.columns):
            self.log_obj.true_log(f'Checking in {feature} column for 0 (zero)')
            df[feature] = df[feature].replace({0.0:None})
                
#             if df[feature].isnull().sum() > 0:
#                 null_dic[feature] = df[feature].isnull().sum()

#         null_df = pd.DataFrame([null_dic], index = ['No. of missing_values']).transpose().reset_index().rename(columns = {'index':'Features'})
        
#         null_df.to_csv('missing_values_data.csv', index = False, header=True)
        
        return df
    
        
    def filling_null_values(self):
        '''In this we will be fill all the null values with the mean.
           For filling out the mean we will be using KNN imputer, where Nearest neighbor will 3.
           This is because, the value which is missing is somehow around its neighbours.
           So we will take the mean of its 3 neighbors and fill the mising values with those
        '''
        df=self.replacing_to_nullvalues()
        self.log_obj.true_log('Defining a fuction which will fill all the null values with mean by using KNNImputer ******')
        
        try:
            self.log_obj.true_log('Creating an object for KNNImputer and we are taking 3 nearest neighbors')
            imputer = KNNImputer(n_neighbors = 3)
            self.log_obj.true_log('Fitting the data')
            impute_data = imputer.fit_transform(df)
            self.log_obj.true_log('''Each sample's missing values are imputed using the mean value from
            3 nearest neighbors found in the data set''')
            self.log_obj.true_log('Creating a separate dataframe that will store the updated data set with mean values in place of null values')
            self.imputed_df = pd.DataFrame(impute_data,columns = df.columns)
            self.log_obj.true_log('Rounding the values such that after decimal there is only two digits present')
            for feature in self.imputed_df.columns:
                self.imputed_df[feature]=self.imputed_df[feature].apply(lambda a: round(a,2))
            self.log_obj.true_log('Rounding complete')
#             self.log_obj.true_log('Saving this dataframe in the final training training file fiolder')
#             self.imputed_df.to_csv('training_data_withoutnull.csv',index=False,header=True)
            return self.imputed_df
        except Exception as e:
            self.log_obj.error_log(f'Unable to fill missing values due to this error: {str(e)}')
                    
        
    def outliers_removal(self):
        self.log_obj.true_log('Defining a function that will remove the outliers from the dataset ******')
        dataframe = self.filling_null_values()
        
        try:
            self.log_obj.true_log('For removing the outlier, quantile function is used from Numpy library')
            
            for i in list(dataframe.columns):
                self.log_obj.true_log(f'For {i} column :')
                q1 = dataframe[i].quantile(0.25)
                q3 = dataframe[i].quantile(0.75)
                self.log_obj.true_log(f'First qunatile position is: {q1}')
                self.log_obj.true_log(f'Third qunatile position is: {q3}')
                
                iqr = q3-q1
                self.log_obj.true_log(f'Inter Quantile Range is {iqr}')
                lower = q1-1.5*iqr
                upper = q3+1.5*iqr
                self.log_obj.true_log(f'Lower range value is: {lower}')
                self.log_obj.true_log(f'Upper range value is: {upper}')
                
                dataframe = dataframe[(dataframe[i] > lower) & (dataframe[i] < upper)]
            self.log_obj.true_log('New dataset after removing the outliers is updated')
            
            return dataframe
        except Exception as e:
            self.log_obj.error_log(f'Unable to remove the out liers because of this error: {str(e)}')

