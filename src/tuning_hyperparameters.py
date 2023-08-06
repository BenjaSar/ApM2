# Imports
import logging
import os
import optuna
from optuna.samplers import TPESampler
import xgboost as xgboost_regressor
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


def pre_processing(pandas_df: pd.DataFrame):
    """This functions is used 
    to preprocess the dataframe gotten after applying 
    the features engineering to the original dataframe.

    Args:
        pandas_df (pd.DataFrame): dataframe obtained after
        loading the dataframe.  

    Returns:
        pd.DataFrame: _description_
    """
    print('Item_MRP', pandas_df['Item_MRP'])
    dataset = pandas_df.drop(columns=['Item_Identifier', 'Outlet_Identifier'])
    print(dataset.info())

    # Split of the dataset in train y test sets
    df_train = dataset.loc[pandas_df['Set'] == 'train']
    df_test = dataset.loc[pandas_df['Set'] == 'test']

    return df_train, df_test


class TuningHyperParametersPipeline(object):
    """_summary_

    Args:
        object (_type_): _description_
    """

    def __init__(self, input_path, output_path: str = None):
        self.input_path = input_path
        self.output_path = output_path

    def load_data(self) -> pd.DataFrame:
        """This function is used to load the dataframe
        to be processed.

        Returns:
            pd.DataFrame: _description_
        """

        try:
            train_file = 'dataframe.csv'
            train_data = os.path.join(self.input_path, train_file)
            pandas_df = pd.read_csv(train_data)
            logging.info("Loading data from:  {self.input_path}")

        except (FileNotFoundError, PermissionError, OSError) as error_load_file:
            logging.exception(
                "An error occurred while loading data: %s", error_load_file)

        return pandas_df

    def model_to_train(self, df: pd.DataFrame):
        """_summary_

        Args:
            df (pd.DataFrame): Dataframe that will be trained.

        Returns:
            trained_model, xval (pd.Dataframe):  The datasets that are 
            gotten after applying machine learning model.
        """

        global x_train
        global y_train

        df_train, df_test = pre_processing(pandas_df=df)

        # Deleting columns without data
        df_train.drop(['Unnamed: 0', 'Set'], axis=1, inplace=True)
        df_test.drop(['Unnamed: 0', 'Item_Outlet_Sales', 'Set'],
                     axis=1, inplace=True)

        seed = 28

        # Splitting of  the dataset in training and validation sets
        X = df_train.drop(columns='Item_Outlet_Sales')
        print('Este es el valor de X')
        X.info()
        y = df_train['Item_Outlet_Sales']

        x_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.3, random_state=seed)
        return x_train, y_train

    def train_model(self, x_train, y_train):
        """_summary_

        Args:
            x_train (np.array): Array to be trained.
            y_train (np.array): Array to be trained.
        """

        seed = 28
        model_trained = xgboost_regressor.XGBRegressor(
            objective='reg:linear', n_estimators=10, random_state=seed)
        # Train the model
        score_model = cross_val_score(
            model_trained, x_train, y_train, scoring='neg_root_mean_squared_error', n_jobs=-1, cv=10)
        print('Score model', score_model)
        print(np.mean(score_model), np.std(score_model))
        return score_model

    def return_score(self, param):
        """_summary_

        Args:
            param (_type_): _description_

        Returns:
            float: metric gotten after of the
            model training.
        """

        model = xgboost_regressor.XGBRegressor(**param)

        root_mean_square_error = np.mean(cross_val_score(
            model, x_train, y_train, cv=4, n_jobs=-1, scoring='neg_root_mean_squared_error'))

        return root_mean_square_error

    def objective(self, trial):
        """_summary_

        Args:
            trial (_type_): _description_
        """
        # Define hyperparameters
        param = {'sampling_method': 'gradient_based', 'reg_lambda':
                 trial.suggest_uniform('lambda', 7.0, 17.0), 'reg_alpha': trial.suggest_uniform('alpha', 7.0, 17.0), 'learning_rate': trial.suggest_uniform('learning_rate', 0.05, 0.5),
                 'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 0.9),
                 'n_estimators': trial.suggest_int('n_estimators', 0, 100)}
        return self.return_score(param)

    def run(self):
        """This function is used to load the trained model, to make the 
        predictions of the model and to write the result in csv file.
        """
        data_frame = self.load_data()
        x_trained, y_trained = self.model_to_train(data_frame)
        model_trained = self.train_model(x_train=x_trained, y_train=y_trained)
        study_object = optuna.create_study(
            direction='minimize', sampler=TPESampler())
        study_object.optimize(self.objective, n_trials=200)


if __name__ == "__main__":

    pipeline = TuningHyperParametersPipeline(
        input_path='../data/',
        output_path='../tuning_hp')
    pipeline.run()
