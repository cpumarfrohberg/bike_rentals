import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#TODO make function for reading in both datasets
transformable = pd.read_csv('./data/train.csv', index_col = 0, parse_dates = True)


PATH = './artifacts/X_fe_col_names.csv'

#TODO think about how to apply everything to both train as well as test
class BikeRentModeler():
    '''Read, split, transform, fit and predict.'''

    def __init__(self, path = PATH) -> None:
        self.path = path

    def prepare_data(self) -> dict:
        '''Returns dict with feature matrix and labels as values.'''
        df = pd.read_csv(self.path, index_col=0, parse_dates=True)
        X = df.drop(['remainder__Client_Status_Post3Months', 'remainder__Client_Status_Post6Months'], axis=1) 
        y = df['remainder__Client_Status_Post3Months']
        return {'feature_matrix': X, 'labels': y}

    def include_timestamps(self, df) -> pd.DataFrame:
        '''Returns DataFrame with time-stamps.'''
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        return df

    def split_timestamp_data(self, X, y) -> dict:
        '''Returns dict consisting of split data (incl. timestamps).'''
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 42)
        X_train_timestamped = self.include_timestamps(X_train)
        X_val_timestamped = self.include_timestamps(X_val)
        return {
            'X_train_fe': X_train_timestamped, 
            'X_val_fe': X_val_timestamped,
            'y_train': y_train,
            'y_val': y_val,
            }

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

    def label_transformer():
        '''Transforms labels to logged vals.'''
        pass

    def bring_back_transformer():
        '''Transforms predictions to unlogged vals.'''
        pass

    def metric_builder():
        '''Calculates root mean square logged error.'''
        pass


