import pytest

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

PATH_INITIAL_DATA = '../data/'
PATH_TRANSFORMED_DATA = '../artifacts/'

#TODO think about how to apply everything to both train as well as test
class BikeRentModeler():
    '''Read, split, transform, fit and predict.'''

    #TODO: check syntax attributes (i.e. drop hard-coded attributes)
    def __init__(self) -> None:
        self.path_initial = PATH_INITIAL_DATA
        self.path_transformed = PATH_TRANSFORMED_DATA

    #TODO: check syntax placeholders; check return statement! 
    def prepare_data(self) -> dict:
        '''Returns dict with feature matrix and labels as values.'''
        loadable = ['Xtrain', 'Xval']
        for data in loadable:
            loaded = list()
            df_train, df_val = pd.read_csv("self.path_initial/'%s'.csv", data, index_col=0, parse_dates=True)
            loaded.append(df_train)
            loaded.append(df_val)
            for data_loaded in loaded:
                X_train, X_val = data_loaded.drop(['casual', 'registered', 'count'], axis=1) 
                y_train, y_val = data_loaded['count']
        return {
            'X_train': X_train, 
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            }

    def include_timestamps(self, df) -> pd.DataFrame:
        '''Returns DataFrame with time-stamps.'''
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        return df

    #TODO: check if split is even necessary (data has already been split!).
    # Also: implement manual split based on dates (<=19th and 19th < dates <=30)?
    # def split_timestamp_data(self, X, y) -> dict:
    #     '''Returns dict consisting of split data (incl. timestamps).'''
    #     X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 42)
    #     X_train_timestamped = self.include_timestamps(X_train)
    #     X_val_timestamped = self.include_timestamps(X_val)
    #     return {
    #         'X_train_fe': X_train_timestamped, 
    #         'X_val_fe': X_val_timestamped,
    #         'y_train': y_train,
    #         'y_val': y_val,
    #         }

    def model_fit(self, X_train, y_train):
        '''Returns fitted Linear ression model.'''
        linreg = LinearRegression(random_state=42) 
        linreg.fit(X_train, y_train)
        return linreg

    def predictions(self, fit_model, X) -> int:
        '''Returns model predictions.'''
        Lin_Reg = fit_model
        return Lin_Reg.predict(X)

    #TODO implement the following methods
    def interpolator():
        '''Fill hourly vals for registered and casual.'''
        pass
    
    def metric_builder(predictions) -> np.array:
        '''Calculates root mean square logged error.'''
        return np.sqrt(mean_squared_log_error(predictions))

    #TODO check np method as well as placeholder
    def label_transformer(y_train, y_val) -> pd.Series:
        '''Transforms labels to logged vals.'''
        labels = [y_train, y_val]
        for label in labels:
            y_train_logged, y_val_logged = np.log1p('%s', label) - 1
        return y_train_logged, y_val_logged

    #TODO check np method as well as placeholder
    def bring_back_transformer(y_train_logged, y_val_logged):
        '''Transforms predictions to unlogged vals.'''
        undone_logged_labels = [y_train_logged, y_val_logged]
        for logged_label in undone_logged_labels:
            y_train_unlogged, y_val_unlogged = np.exp('%s', logged_label)
        return y_train_unlogged, y_val_unlogged



