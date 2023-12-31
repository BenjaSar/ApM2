"""
feature_engineering.py

DESCRIPTION:
AUTHORS:
FS
Date: July 19th 2023
"""

# Imports
import os
import pandas as pd


class FeatureEngineeringPipeline(object):
    """This is a function to generate the feature 
    engineering and the EDA proccess

    Args:
        object (_type_): data to be transformed
    """

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """This is a function to read data from the
        the data folder.

        Returns:
            pd.DataFrame: The desired DataLake table as a DataFrame.
        """
        try:
            train_file = 'Train_BigMart.csv'
            train_data = os.path.join(self.input_path, train_file)
            data_train = pd.read_csv(train_data)
            data_train['Set'] = 'train'
            test_file = 'Test_BigMart.csv'
            test_data = os.path.join(self.input_path, test_file)
            data_test = pd.read_csv(test_data)
            data_test['Set'] = 'test'

            pandas_df = pd.concat([data_train, data_test],
                                  ignore_index=True, sort=False)

        except FileNotFoundError:
            msg = 'No such file or directory'
            print(msg)

        return pandas_df

    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """This function in used to transform the data and apply EDA 
        process.

        Args:
            df (pd.DataFrame): Dataframe to be transformed.

        Returns:
            pd.DataFrame: Dataframe which was processed by the function.
        """

        df.describe()
        df['Outlet_Establishment_Year'] = 2020 - \
            df['Outlet_Establishment_Year']

        # Unify labels for "Item_Fat_Content "
        df['Item_Fat_Content'] = df['Item_Fat_Content'].replace(
            {'low fat':  'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

        # Verification of unify labels
        set(df['Item_Fat_Content'])

        # LIMPIEZA:  Missing data of Item_Weight variables
        productos = list(df[df['Item_Weight'].isnull()]
                         ['Item_Identifier'].unique())
        for producto in productos:
            moda = (df[df['Item_Identifier'] == producto]
                    [['Item_Weight']]).mode().iloc[0, 0]
            df.loc[df['Item_Identifier'] == producto, 'Item_Weight'] = moda

        # CLEANING DATA: Missing data on Outlet_Size variables
        outlets = list(df[df['Outlet_Size'].isnull()]
                       ['Outlet_Identifier'].unique())

        for outlet in outlets:
            df.loc[df['Outlet_Identifier'] == outlet, 'Outlet_Size'] = 'Small'

        # New category item fat
        df.loc[df['Item_Type'] == 'Non perishable', 'Item_Fat_Content'] = 'NA'

        # FEATURES ENGINEERING: Generating categories for 'Item_Type'
        df['Item_Type'] = df['Item_Type'].replace({'Others': 'Non perishable',
                                                   'Health and Hygiene': 'Non perishable', 'Household': 'Non perishable',
                                                   'Seafood': 'Meats', 'Meat': 'Meats', 'Baking Goods': 'Processed Foods',
                                                   'Frozen Foods': 'Processed Foods', 'Canned': 'Processed Foods', 'Snack Foods': 'Processed Foods', 'Breads': 'Starchy Foods', 'Breakfast': 'Starchy Foods', 'Soft Drinks': 'Drinks', 'Hard Drinks': 'Drinks', 'Dairy': 'Drinks'})

        # Codificación de los precios de productos
        print(pd.qcut(df['Item_MRP'], 4,).unique())
        df['Item_MRP'] = pd.qcut(df['Item_MRP'], 4, labels=[1, 2, 3, 4])

        # Codificación de variables ordinales
        data = df.drop(columns=['Item_Type', 'Item_Fat_Content']).copy()

        data['Outlet_Size'] = data['Outlet_Size'].replace(
            {'High': 2, 'Medium': 1, 'Small': 0})
        data['Outlet_Location_Type'] = data['Outlet_Location_Type'].replace(
            {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0})
        print(data.head(5))

        # Codificación de variables nominales
        data_transformed = pd.get_dummies(
            data, columns=['Outlet_Type'], dtype=int)
        data_transformed.info()

        return data_transformed

    def write_prepared_data(self, transformed_df: pd.DataFrame) -> None:
        """This function write the data transforme to the csv file.

        Args:
            transformed_df (pd.DataFrame): This data is gotten during EDA
            process.

        Returns:
            transformed_df (pd.Dataframe): _description_
        """
        try:
            name_file = 'dataframe.csv'
            output_file = os.path.join(self.output_path, name_file)
            transformed_df.to_csv(output_file)
        except (IOError, OSError):
            print("Error writing to file")

        return None

    def run(self):
        """This functios is used to read, transform and 
        write to csv file the transformed data."""
        data_frame = self.read_data()
        df_transformed = self.data_transformation(data_frame)
        self.write_prepared_data(df_transformed)


if __name__ == "__main__":
    FeatureEngineeringPipeline(input_path='../data/',
                               output_path='../data/').run()
