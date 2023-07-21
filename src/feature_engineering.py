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
        input_path = '././data/'

        data_train = pd.read_csv(input_path + 'Train_BigMart.csv')
        data_test = pd.read_csv(input_path + 'Test_BigMart.csv')
        pandas_df = pd.concat([data_train, data_test], ignore_index=True, sort=False)
        print(pandas_df.head(20))

        return pandas_df

    
    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        COMPLETAR DOCSTRING
        
        """
        
        df.describe()
        df['Outlet_Establishment_Year'] = 2020 - df['Outlet_Establishment_Year']

        #Unify labels for "Item_Fat_Content "
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})
        
        # Verificamos la unificación de etiquetas
        set(df['Item_Fat_Content'])

        #LIMPIEZA: faltante peso productos
        productos = list(df[df['Item_Weight'].isnull()]['Item_Identifier'].unique())
        for producto in productos:
            moda = (df[df['Item_Identifier'] == producto][['Item_Weight']]).mode().iloc[0,0]
            df.loc[df['Item_Identifier'] == producto, 'Item_Weight'] = moda

        # LIMPIEZA: faltante en el tamaño de las tiendas
        outlets = list(df[df['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())

        for outlet in outlets:
            df.loc[df['Outlet_Identifier'] == outlet, 'Outlet_Size'] =  'Small'

        #New category item fat
        df.loc[df['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

        # FEATURES ENGINEERING: creando categorías para 'Item_Type'
        df['Item_Type'] = df['Item_Type'].replace({'Others': 'Non perishable', 'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
                                                       'Seafood': 'Meats', 'Meat': 'Meats','Baking Goods': 'Processed Foods', 
                                                       'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods',
                                                       'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods',
                                                       'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})


        df['Item_Type'].unique()
        
        
        return df_transformed

   # def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        
        # COMPLETAR CON CÓDIGO
        
        #return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
     #   self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path = '../data/', 
                               output_path = '../data/dataTest.csv').run()