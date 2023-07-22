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

    
    #def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """
        
        # COMPLETAR CON CÓDIGO
        
        return df_transformed

    #def model_dump(self, model_trained) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        
        # COMPLETAR CON CÓDIGO
        
        return None

    def run(self):
    
        df = self.read_data()
    #    model_trained = self.model_training(df)
    #    self.model_dump(model_trained)

if __name__ == "__main__":

    ModelTrainingPipeline(input_path = './data/',
                          model_path = '../model').run()