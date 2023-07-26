"""
train.py

DESCRIPCTION: In this file you will find the functions
to train the model and write the result in a csv file
AUTHORS:
FS
Vilcamiza Espinoza, Gerardo Alexis
Date: July 22rd 2023
"""

# Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle


def write_model(dataframe_train: pd.DataFrame, dataframe_test: pd.DataFrame):
    """This function receive the dataset for training and testing 

    Args:
        dataframe_train (pd.Dataframe): Dataframe for training
        dataframe_test (pd.Dataframe): Datafrane for testing

    Returns:
        csv files: train and test files in csv format 
    """
    try:
        out_path = '../model'
        train_file = 'train_final.csv'
        test_file = 'test_final.csv'
        output_train = os.path.join(out_path, train_file)
        output_test = os.path.join(out_path, test_file)
        dataframe_train.to_csv(output_train)
        print(f'Writing  dataframe train...')
        dataframe_test.to_csv(output_test)
        print(f'Writing dataframe test...')

    except Exception as error:
        print("An exception has ocurred: ", type(error).__name__, "-", error)

    return train_file, test_file


class ModelTrainingPipeline(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """This function read the dataframe from train file.

        Returns:
            pd.DataFrame> The desired datalake table as Dataframe
        """
        try:
            train_file = 'dataframe.csv'
            train_data = os.path.join(self.input_path, train_file)
            pandas_df = pd.read_csv(train_data)

            print(pandas_df.head(20))

        except Exception as error:
            print("An exception has ocurred: ",
                  type(error).__name__, "-", error)

        return pandas_df

    def model_training(self, df: pd.DataFrame):
        """_summary_

        Args:
            df (pd.DataFrame): Dataframe that will be trained.

        Returns:
            trained_model, xval (pd.Dataframe):  The datasets that are gotten  after 
            apply machine learning model.
        """
        dataset = df.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

        # Split of the dataset in train y test sets
        df_train = dataset.loc[df['Set'] == 'train']
        df_test = dataset.loc[df['Set'] == 'test']

        # Deleting columns without data
        df_train.drop(['Unnamed: 0', 'Set'], axis=1, inplace=True)
        print(df_train.head(5))
        df_test.drop(['Unnamed: 0', 'Item_Outlet_Sales', 'Set'],
                     axis=1, inplace=True)

        # Writing the model in a file
        write_model(dataframe_train=df_train, dataframe_test=df_test)

        seed = 28
        model = LinearRegression()

        # Splitting of  the dataset in training and validation sets
        X = df_train.drop(columns='Item_Outlet_Sales')
        x_train, x_val, y_train, y_val = train_test_split(
            X, df_train['Item_Outlet_Sales'], test_size=0.3, random_state=seed)

        # Training the model
        trained_model = model.fit(x_train, y_train)

        return trained_model, x_val

    def model_dump(self, model_trained) -> None:
        """_summary_

        Args:
            model_trained (.pkl): _description_

        Returns:
            _type_: _description_
        """
        trained_file = "trained_model.pkl"
        model_output = open(os.path.join(self.model_path, trained_file), 'wb')
        print(f'Writing pickle file....')
        pickle.dump(model_trained, model_output)
        model_output.close()
        print(f'The pickle file was written.')

        return None

    def run(self):

        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)


if __name__ == "__main__":

    ModelTrainingPipeline(input_path='../data/',
                          model_path='../model').run()
