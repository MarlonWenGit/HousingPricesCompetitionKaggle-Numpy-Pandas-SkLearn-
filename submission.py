# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn.model_selection import train_test_split

def get_home_data(csv_name):
    data_path = f"/kaggle/input/house-prices-advanced-regression-techniques/{csv_name}.csv"
    return pd.read_csv(data_path)

def get_data(csv_name):
    # Get the csv table data
    home_data = get_home_data(csv_name)
    
    # What attributes to use to train the model
    features = ['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
           'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
           'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
           'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
           'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
           'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
           'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
           'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
           'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
           'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
           'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
           'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
           'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
           'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
           'SaleCondition']
    X = home_data[features]
    X = clean_data(X)

    if csv_name == "train":
        y = home_data.SalePrice
        return X, y
    return X

def clean_data(X):
    # Convert string values to floats
    X = pd.get_dummies(X)
    
    # Fill missing values
    X = X.fillna(0)

    return X

X, y = get_data('train')

# Split into data used to train and test the model
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes):
    '''
    Gets the mean absolute error taking the max_leaf_nodes as a parameter
    '''
    housing_model = RandomForestRegressor(
        random_state=1,
        n_estimators=500,
        max_depth=20,
        min_samples_leaf=3,
        max_features="sqrt",
        max_leaf_nodes=max_leaf_nodes
    )
    housing_model.fit(train_X, train_y)
    predictions = housing_model.predict(val_X)
    return mean_absolute_error(predictions, val_y)

# Iterate through possible values and try to find the one that gives the minimum absolute error
# Iteration 1: {25: 21971.17169727153, 50: 19931.416378568178, 100: 18747.375985493312, 200: 18412.582755308966, 400: 18412.582755308966, 800: 18412.582755308966}
# Iteration 2: {200: 18412.582755308966, 250: 18412.582755308966, 300: 18412.582755308966, 350: 18412.582755308966, 400: 18412.582755308966}
# 200 is the optimal value
candidate_max_leaf_nodes = [200, 50, 100, 200, 400]
cDict = {leaf_nodes: get_mae(leaf_nodes) for leaf_nodes in candidate_max_leaf_nodes}
print(cDict)
best_max_leaf_nodes = min(cDict, key=cDict.get)

housing_model = RandomForestRegressor(
    random_state=1,
    n_estimators=500,
    max_depth=20,
    min_samples_leaf=3,
    max_features="sqrt",
    max_leaf_nodes=200
)
housing_model.fit(train_X, train_y)
predictions = housing_model.predict(val_X)

# Check sample submission for columns
get_home_data('sample').describe()

# Submit predictions to Kaggle
submission = pd.DataFrame({
    "Id": get_home_data('test')["Id"],
    "SalePrice": predictions
})
submission.to_csv("my_submission.csv", index=False)

