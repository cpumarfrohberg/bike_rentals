import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

#%% reading & inspecting data
df = read_csv("trainBikes.csv", ",")

#%% EDA
# p1: distribution of "count"
# p2: plotting of "count" with other features
# 
# 
# 

#%% Train-test split

#%% FE - creating new feature re daily sales "peaks"


#%% FE - applying function to selected feature(s)

#%% FE - onehotencoding?

#%% FE - KbinsDiscretizer?


#%% FE - scaling













