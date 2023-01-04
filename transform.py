from utils import transformable, ChurnModeler

import pickle, time, logging

import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn import set_config
set_config(transform_output = 'pandas')
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.compose import make_column_transformer

logging.basicConfig(level = logging.DEBUG)

logging.info('prep data for transformation')
discretizable = ['Client_Age_Years', 'n(Loans)_Outstanding_Maynas', 'n(Loans)_Outstanding_Maynas',
                'n(Loans)_Outstanding_Other', 'n(Additional_Loans)_Post3Months', 
                'n(Additional_Loans)_Post6Months','n(Additional_Loans)_Pre3Months', 
                'n(Additional_Loans)_Pre6Months']

min_max_scalable = ['n(Months)_Since_Last_Disbursement', 'n(Months)_Client_Relationship',
            'n(Months)_LO_Active_Employee', 'Total_Accumulated_Interest_per_Client', 
            'n(Months)_Change_LO']

standard_scalable = ['Amount_Last_Disbursement']

discretizer = KBinsDiscretizer(encode='onehot-dense')
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

transformable.drop('Client_ID', axis = 1, inplace = True)

if __name__ == '__main__':
    churn_prepper = ChurnModeler()
    time.sleep(1)
    logging.debug('creating month and time cols')
    X_fe_time = churn_prepper.include_timestamps(transformable)
    
    time.sleep(1)
    logging.debug('transforming')
    preprocessor = make_column_transformer( 
            (discretizer, discretizable),
            (min_max_scaler, min_max_scalable),
            (standard_scaler, standard_scalable),
            remainder='passthrough'
            )
    X_feature_engineered = preprocessor.fit_transform(X_fe_time)

    time.sleep(1)
    logging.debug(f'Inserting col names.')
    X_fe_col_names = pd.DataFrame(X_feature_engineered)

    logging.info(f'transformation concluded: X_fe_col_names created. \
                Shape: {X_fe_col_names.shape}. Columns: {X_fe_col_names.columns}')

    time.sleep(1)
    logging.debug('saving X_fe_col_names as .csv-file')
    X_fe_col_names.to_csv('./artifacts/X_fe_col_names.csv')
 
    time.sleep(2)
    logging.info('X_fe_col_names.csv saved')




