#transform.py
'''Transform data.
Based on GridSearch, use best model for fit on transformed data.'''

import time, logging
logging.basicConfig(level = logging.DEBUG)

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils import (best_model_identifier, datasets, PARAMS)

def main():
        time.sleep(1)
        logging.debug('load transformed data')
        datasets_fillable = list()
        dataset = [pd.read_csv(f'./artifacts/{element}.csv') for element in datasets]
        datasets_fillable.append(dataset)
        X_train_fe = datasets_fillable[0]
        #TODO: call crossvalidation in transform.py!

        time.sleep(1)
        #TODO: logging.debug('identify best model')
        pass

if __name__ == '__main__':
        main()




