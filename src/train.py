"""
train.py

COMPLETAR DOCSTRING

DESCRIPCIÓN:
AUTOR:
FECHA:
"""

# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import datetime as dt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomTreesEmbedding
from scipy import stats

class ModelTrainingPipeline(object):

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING 
        
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
            
        try:
            input_path = './data/'
            train_file = 'dataframe.csv'
            train_data = os.path.join(input_path, train_file)
            pandas_df = pd.read_csv(train_data)
            
            print(pandas_df.head(20))

        except Exception as error: 
            print("An exception ocurred: ", type(error).__name__,"-", error) 
        
        return pandas_df

    
    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """
        dataset = df.drop(columns=['Item_Identifier', 'Outlet_Identifier']) 

        #Division del dataset en train y test
        df_train = dataset.loc[df['Set'] == 'train']
        df_test  = dataset.loc[df['Set'] == 'test']

        # Eliminando columnas sin datos
        df_train.drop(['Unnamed: 0','Set'], axis=1, inplace=True)
        print(df_train.head(5))
        df_test.drop(['Unnamed: 0', 'Item_Outlet_Sales','Set'], axis=1, inplace=True)

        try:
            out_path = './model'
            train_file = 'train_final.csv'
            test_file  = 'test_final.csv'
            output_train = os.path.join(out_path, train_file)
            output_test = os.path.join(out_path, test_file)
            df_train.to_csv(output_train)
            df_test.to_csv(output_test)
        except Exception as error: 
            print("An exception ocurred: ", type(error).__name__,"-", error) 
        
        #return df_transformed
        return df_train

    #def model_dump(self, model_trained) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        
        # COMPLETAR CON CÓDIGO
        
        return None

    def run(self):
    
        df = self.read_data()
        model_trained = self.model_training(df)
    #    self.model_dump(model_trained)

if __name__ == "__main__":

    ModelTrainingPipeline(input_path = './data/',
                          model_path = '../model').run()