#transform.py
'''Transform data.
Based on GridSearch, use best model for fit on transformed data.'''
import warnings
warnings.filterwarnings("ignore")

import time, logging
logging.basicConfig(level = logging.DEBUG)

from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, KBinsDiscretizer)
from sklearn.compose import make_column_transformer

from utils import (read_for_split, include_timestamps)

oh_encodables = ['holiday', 'workingday']
binnables = ['Hour', 'Month']
min_max_scalables = ['weather', 'temp', 'humidity', 'windspeed']
#TODO: sum vals from 'count', 'registered' and 'casual' and create a new label (being the sum)

oh_encoder = OneHotEncoder(handle_unknown = 'ignore', drop = 'first')#.set_output(transform = 'pandas')
binner = KBinsDiscretizer(encode='onehot-dense')#.set_output(transform = 'pandas')
min_max_scaler = MinMaxScaler()#.set_output(transform = 'pandas')

def main():
        '''Import, transform and save data.'''
        time.sleep(1)
        logging.debug("create 'Hour' and 'Month' cols")
        X_train_dict = read_for_split()
        X_fe_time = include_timestamps(X_train_dict.get('feature_matrix'))
        logging.debug(f'created X_fe_time with the following cols :{X_fe_time.info()} \
                        and shape: {X_fe_time.shape}')

        time.sleep(1)
        logging.debug('transforming')
        preprocessor = make_column_transformer(
                (oh_encoder, oh_encodables),
                (binner, binnables),
                (min_max_scaler, min_max_scalables),
                remainder='passthrough'
                )        
        X_feature_engineered = preprocessor.fit_transform(X_fe_time)
        logging.debug(f'created X_fe_time with the following cols :{X_fe_time.columns} \
                        and shape: {X_fe_time.shape} and the following datatypes: \
                        {X_fe_time.info()}')

        time.sleep(3)
        logging.debug('saving X_feature_engineered as .csv-file')
        X_feature_engineered.to_csv('./artifacts/train_feature_engineered.csv')

        time.sleep(2)
        logging.info('X_feature_engineered.csv saved')

if __name__ == '__main__':
        main()