#transform.py
'''Transform data.
Based on GridSearch, use best model for fit on transformed data.'''
import warnings
warnings.filterwarnings("ignore")

import time, logging
logging.basicConfig(level = logging.DEBUG)

import pandas as pd

from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, KBinsDiscretizer)
from sklearn.compose import make_column_transformer

from utils import (read_for_split, split_data, include_timestamps)

oh_encodables = ['holiday', 'workingday', 'season']
binnables = ['Hour', 'Month']
min_max_scalables = ['weather', 'temp', 'humidity', 'windspeed']

oh_encoder = OneHotEncoder(handle_unknown = 'ignore', drop = 'first')#.set_output(transform = 'pandas')
binner = KBinsDiscretizer(encode='onehot-dense')#.set_output(transform = 'pandas')
min_max_scaler = MinMaxScaler()#.set_output(transform = 'pandas')

preprocessor = make_column_transformer(
        (oh_encoder, oh_encodables),
        (binner, binnables),
        (min_max_scaler, min_max_scalables),
        remainder='passthrough'
        )        

#TODO: sum vals from 'count', 'registered' and 'casual' and create a new label (being the sum)
#TODO: save labels y_train by dropping DateTime

def main():
        '''Import, transform and save data.'''
        time.sleep(1)
        logging.debug("create 'Hour' and 'Month' cols")
        X_train_dict = read_for_split()

        split_data_dict = split_data(X_train_dict['feature_matrix'], X_train_dict['labels'])

        logging.debug(f'Split data and created dict with following keys: {split_data_dict.keys()}')

        X_train, X_val, y_train, y_val = split_data_dict.get('X_train'), split_data_dict.get('X_val'), split_data_dict.get('y_train'), split_data_dict.get('y_val')
        
        
        time.sleep(3)
        logging.info(f'Sizes of split data: X_train: {X_train.shape}, \
                X_val: {X_val.shape}, y_train: {y_train.shape}, y_val: {y_val.shape}')
        
        time.sleep(1)
        logging.info('Creating month and time cols for X_train and X_val')
        X_train_time = include_timestamps(X_train)
        X_val_time = include_timestamps(X_val)

        time.sleep(1)
        logging.debug('transforming')
        X_train_fe = preprocessor.fit_transform(X_train_time)
        X_train_fe = pd.DataFrame(X_train_fe, columns = preprocessor.get_feature_names_out())
        # logging.debug(f'created X_train_fe with the following cols :{X_train_fe.columns} \
        #                 and shape: {X_train_fe.shape} and the following datatypes: \
        #                 {X_train_fe.info()}')
        
        X_val_fe = preprocessor.fit_transform(X_val_time)
        X_val_fe = pd.DataFrame(X_val_fe, columns = preprocessor.get_feature_names_out())
        logging.debug(f'created X_val_fe with the following cols :{X_val_fe.columns} \
                        and shape: {X_val_fe.shape} and the following datatypes: \
                        {X_val_fe.info()}')
        
        time.sleep(1)
        logging.info(f'Dropping timestamps of y_train and y_val.')
        y_train = y_train.to_frame()
        y_val = y_val.to_frame()
        y_train.reset_index(inplace = True)
        logging.info(f'Cols of y_train: {y_train.columns} and y_val: {y_val.columns}.')
        y_train.drop('datetime', axis = 1, inplace = True)
        y_val.reset_index(inplace = True)
        y_val.drop('datetime', axis = 1, inplace = True)
        logging.info(f'Shapes of y_train: {y_train.shape} and y_val: {y_val.shape}.')
        
        time.sleep(2)
        logging.debug('Saving transformed data to .csv-files')
        data_for_model_fit = [ X_val_fe, X_val_fe, y_train, y_val]
        datasets = ['X_train_transformed', 'X_val_transformed', 'y_train', 'y_val']
        for dataset_name, frame in zip(datasets, data_for_model_fit):
                frame.to_csv(f'./artifacts/{dataset_name}.csv')

if __name__ == '__main__':
        main()