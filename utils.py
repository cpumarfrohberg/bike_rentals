'''Class for EDA as well as for making preds.
@attributes:
    - path variable for reading in initial datasets (Xtrain.csv and Xval.csv)
    - path variable for reading in transformed dataset (additional transformations to timestamps: tbd)

@methods:
    - ".read_for_x()" => loaders of datasets: one for EDA, the other for transformed dataset
    - ".include_timestamps()" => first transformer: include timestamps
        => TODO: include hourly vals (especially since current values are not given for a specific time period)
        => TODO: think of additional FE (e.g. KBinsDiscretizer() or PolynomialFeatures(); note that best submissions to Kaggle don't include that sort of FE...
    - ".model_fit()": fit a vanilla LinReg()
        => TODO: do a GridSearchCV() and create pipeline based on selected best params        
        => TODO: call .make_pipeline(), combining CV with make_pipeline() so as to avoid leakage        
    - ".predictions()": make predictions on logged labels
    - ".bring_back_transformer()": unlog predictions
    - ".goodness_fit()": rmsle of predictions
    => TODO: check how to implement ".interpolator()"
    '''





import pytest

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

PATH_INITIAL_DATA = '../data/'
PATH_TRANSFORMED_DATA = '../artifacts/'

#TODO think about how to apply everything to both train as well as test
class BikeRentPredictor():
    '''Read, split, transform, fit and predict.'''

    #TODO: check syntax attributes (i.e. drop initializer?)
    def __init__(self) -> None:
        self.path_initial = PATH_INITIAL_DATA
        self.path_transformed = PATH_TRANSFORMED_DATA

    #TODO: check syntax for "parametrized dataload" (i.e. placeholder)
    def read_for_EDA(self) -> pd.DataFrame:
        '''Returns dict with feature matrix and labels as values.'''
        loadable = ['Xtrain', 'Xval']
        for data in loadable:
            df_train, df_val = pd.read_csv("self.path_initial/'%s'.csv", data, index_col=0, parse_dates=True)
            concatenated = pd.concat(df_train, df_val)
        return concatenated

    #TODO:  check if empty list is even necessary! statement check syntax placeholders! 
    def read_for_split(self) -> dict:
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

    #TODO implement the following methods
    def interpolator():
        '''Fill hourly vals for registered and casual.'''
        pass

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

    #TODO check np method as well as placeholder
    def label_transformer(y_train, y_val) -> pd.Series:
        '''Transforms labels to logged vals.'''
        labels = [y_train, y_val]
        for label in labels:
            y_train_logged, y_val_logged = np.log1p('%s', label) 
        return y_train_logged, y_val_logged

    def model_fit(self, X_train, y_train_logged) -> LinearRegression():
        '''Returns fitted Linear Regression (transformed labels).'''
        linreg = LinearRegression(random_state=42) 
        fit_model = linreg.fit(X_train, y_train_logged)
        return fit_model

    def predictions(self, fit_model, X_val) -> np.array:
        '''Returns model predictions
        Takes Xval'''
        predictions_logged = fit_model.predict(X_val) 
        return predictions_logged

    #TODO check following two np methods
    def bring_back_transformer(predictions_logged) -> pd.Series:
        '''Transforms predictions to unlogged vals.'''
        predictions_unlogged = np.exp(predictions_logged) - 1
        return predictions_unlogged
        
    def goodness_fit(predictions) -> np.array:
        '''Calculates root mean square logged error.'''
        return np.sqrt(mean_squared_log_error(predictions))


