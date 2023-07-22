"""
feature_engineering.py

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
from scipy import stats


class FeatureEngineeringPipeline(object):
    """This is a function to generate the feature 
    engineering and the EDA proccess

    Args:
        object (_type_): _description_
    """

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """_summary_

        Returns:
            pd.DataFrame: _description_
        """
        
        """
        COMPLETAR DOCSTRING 
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """

        try:
            input_path = './data/'
            train_file = 'Train_BigMart.csv'
            train_data = os.path.join(input_path, train_file)
            data_train = pd.read_csv(train_data)
            test_file = 'Test_BigMart.csv'
            test_data = os.path.join(input_path, test_file)
            data_test = pd.read_csv(test_data)
            pandas_df = pd.concat([data_train, data_test], ignore_index=True, sort=False)
            print(pandas_df.head(20))

        except Exception as error: 
            print("An exception ocurred: ", type(error).__name__,"-", error) 
        
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
       
        #Codificación de los precios de productos
        df['Item_MRP'] = pd.qcut(df['Item_MRP'], 4, labels = [1, 2, 3, 4])

        #Codificación de variables ordinales
        data = df.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()        
        
        data['Outlet_Size'] = data['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
        data['Outlet_Location_Type'] = data['Outlet_Location_Type'].replace({'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0}) 

        #Codificación de variables nominales
        data_transformed = pd.get_dummies(data, columns=['Outlet_Type'], dtype=int)
        print(data.head(6))
        
        
        return data_transformed

    def write_prepared_data(self, transformed_df: pd.DataFrame) -> None:
        """
        COMPLETAR DOCSTRING
        
        """
        
        # COMPLETAR CON CÓDIGO
        try:
            out_path = './data'
            name_file = 'dataframe.csv'
            output_file = os.path.join(out_path, name_file)
            transformed_df.to_csv(output_file)
        except Exception as error: 
            print("An exception ocurred: ", type(error).__name__,"-", error) 
            
        return None

    def run(self):
    
        df = self.read_data()
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)

  
if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path = './data/', 
                               output_path = './data/').run()