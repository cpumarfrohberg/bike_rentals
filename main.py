#main.py
'''Load, transform and fit - long version.'''

import pickle, time, logging
from warnings import filterwarnings
filterwarnings(action='ignore')

import pandas as pd

logging.basicConfig(level = logging.DEBUG)

from utils import BikeRentPredictor

def main():
    #TODO: load transformed dataset
    split_dict = model_data.split_data(X = prepped_data['feature_matrix'], 
                                                y = prepped_data['labels'])
    X_train, X_val, y_train, y_val = (split_dict['X_train'], split_dict['X_val'], 
                                    split_dict['y_train'], split_dict['y_val'])
    
    time.sleep(2)
    logging.debug('model fit on X_train and X_val')
    lin_reg = model_data.model_fit(
        X_train = X_train, 
        y_train = y_train,
        )
    
    time.sleep(2)
    logging.debug('making predictions on X_train and X_val')
    pred_X_train = model_data.predictions(
        fit_model = lin_reg,
        X = X_train
        )

    pred_X_val = model_data.predictions(
        fit_model = lin_reg,
        X = X_val
         )
    
    rmsle_train = model_data.goodness_fit(y_train, pred_X_train).round(2)
    rmsle_val = model_data.goodness_fit(y_val, pred_X_val).round(2)

    time.sleep(2)
    logging.info(f'The rmsle - score based on training set is: {(rmsle_train).round(2)}')
    time.sleep(2)
    logging.info(f'The rmsle - score based on validation set is: {(rmsle_val).round(2)}')

    time.sleep(1)   
    logging.info('saving full model')
    #TODO: call only .to_csv()?
    with open('./artifacts/churn-model-refactored.bin', 'wb') as f_out:
        pickle.dump(lin_reg, f_out) 
    time.sleep(2)   
    logging.info('full model saved as churn-model-refactored.bin')

if __name__ == '__main__':
    main()
    
    

    




