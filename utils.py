#utils.py
'''Build helper functions for transformation, model fit and making predictions.
vs. 230123:
    - apply ColumnTransformer for 
        -> creating time-related columns 
        -> KBinsDiscretize yearly cols
        -> OHE cats
        => save transformed dataset
    - apply CV with: Polynomial Regressor, Random Forest Regressor and Poisson Regressor
'''
import pytest

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV #TODO: evaluate with RandomizedGridSearch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error

datasets = ['X_train_transformed', 'X_val_transformed', 'y_train', 'y_val']

PARAMS = {
    'models' : ['LinearRegression', 'RandomForestRegressor', 'PoissonRegressor'],
    'polynomial_degrees' : [2, 3, 5],
    'splits' : [3, 5],
    'trees' : [30, 100, 300],
    'max_depth' : [3, 5, 10]
}

def best_model_identifier(model, params, splits, n_jobs, X_train, y_train, score):
    '''Return mean r2-score for optional models.
    Takes untransformed dataset, transforms X_val in each fold and 
    returns score.'''
    validator = GridSearchCV(
        estimator = model, 
        param_grid = params, 
        cv = splits, 
        n_jobs = n_jobs,
        scoring = score) 
    validator.fit(X_train, y_train)
    best_model = validator.best_estimator_ 
    return best_model

def read_for_split() -> dict:
    '''Returns dict with feature matrix and labels as values.'''
    X = pd.read_csv('./data/train.csv', index_col=0, parse_dates=True)
    y = X['count']
    X.drop(['casual', 'registered', 'count'], axis=1, inplace=True) 
    return {'feature_matrix': X, 'labels': y}

def split_data(X, y) -> dict:
        '''Returns dict consisting of split data (incl. timestamps).'''
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 42)
        return {
            'X_train': X_train, 
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            }

def include_timestamps(df) -> pd.DataFrame:
    '''Returns DataFrame with time-stamps.'''
    df['Hour'] = df.index.hour
    df['Month'] = df.index.month
    return df

def label_transformer(y_train, y_val) -> pd.Series:
    '''Transforms labels to logged vals.'''
    labels = [y_train, y_val]
    for label in labels:
        y_train_logged, y_val_logged = np.log1p(f'{label}') 
    return y_train_logged, y_val_logged

def model_fit(X_train, y_train_logged) -> LinearRegression():
    '''Returns fitted Linear Regression (transformed labels).'''
    linreg = LinearRegression(random_state=42) 
    fit_model = linreg.fit(X_train, y_train_logged)
    return fit_model

def predictions(fit_model, X_val) -> np.array:
    '''Returns model predictions
    Takes Xval'''
    predictions_logged = fit_model.predict(X_val) 
    return predictions_logged

def bring_back_transformer(predictions_logged) -> pd.Series:
    '''Transforms predictions to unlogged vals.'''
    predictions_unlogged = np.exp(predictions_logged) - 1
    return predictions_unlogged
    
def goodness_of_fit(predictions) -> np.array:
    '''Calculates root mean square logged error.'''
    return np.sqrt(mean_squared_log_error(predictions))


def test_read_for_split():
    data_loaded = read_for_split()
    assert type(data_loaded) == dict