"""
feature_engineering.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports

import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import datetime as dt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding
from scipy import stats


class FeatureEngineeringPipeline(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING 
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
        input_path = '../data'

        data_train = pd.read_csv(input_path + '/Train_BigMart.csv')
        data_test = pd.read_csv(input_path + '/Test_BigMart.csv')
        pandas_df = pd.concat([data_train, data_test], ignore_index=True, sort=False)
        print(pandas_df.head(20))

        return pandas_df

    
    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """
        df['Outlet_Establishment_Year'] = 2020 - df['Outlet_Establishment_Year']
        
        
        return df_transformed

   # def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        
        # COMPLETAR CON CÓDIGO
        
        return None

    def run(self):
    
        df = self.read_data()
        self.write_prepared_data(panda_df)
     #   df_transformed = self.data_transformation(df)
     #   self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path = '../data/Train_BigMart.csv',
                               output_path = '../data/Mi/dataTest').run()