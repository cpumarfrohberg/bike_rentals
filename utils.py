import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

transformable = pd.read_csv('./data/Tabla_01_English_Unique_postEDA.csv', index_col = 0, parse_dates = True)
PATH = './artifacts/X_fe_col_names.csv'
WEIGHTS = {0:0.41, 1:0.59}

class ChurnModeler():
    '''Read, transform, fit and predict.'''

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
    
    def split_data(self, X, y) -> dict:
        '''Returns dict consisting of split data (incl. timestamps).'''
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 42)
        return {
            'X_train': X_train, 
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            }

    def model_fit(self, X_train, y_train, weights):
        '''Returns fitted Logistic Regression model.'''
        clf_LR = LogisticRegression(class_weight = weights, random_state=42) 
        clf_LR.fit(X_train, y_train)
        return clf_LR

    def predictions(self, fit_model, X):
        '''Returns model predictions.'''
        clf = fit_model
        return clf.predict(X)

def test_sparsity():
    assert [(transformable.to_numpy() == 0) > 0.5]

