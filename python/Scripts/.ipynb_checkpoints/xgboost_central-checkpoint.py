import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost as xgb
import seaborn as sns

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir) 

from Scripts.datagen import Datagen

SEED = 42


class Xgboost(object):
    def __init__(self):
        data = Datagen()
        df1 = data.Gen1
        df2 = data.Gen2
        df3 = data.Gen3
        df4 = data.Gen4       
        df5 = data.Gen5
        
        print("..........training for Generator 1.........")
        self.train_model(df1, "Generator1")
        print("..........training for Generator 2.........")
        self.train_model(df2, "Generator2")
        print("..........training for Generator 3.........")
        self.train_model(df3, "Generator3")
        print("..........training for Generator 4.........")
        self.train_model(df4, "Generator4")
        print("..........training for Generator 5.........")
        self.train_model(df5, "Generator5")
        
        
    def train_model(self, df, fname):
        # Splitting the training and testing data
        self.train = df.loc[df.index < '01-07-2016']
        #self.train = df
        self.test = df.loc[df.index >= '01-08-2016']
        
        # Defining training and testing data
        # X_train = self.train[['hour', 'dayofweek', 'month', 'quarter', 'year', 'dayofyear']]
        X_train = self.train[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        y_train = self.train[list(df)[0]]

        # X_test = self.test[['hour', 'dayofweek', 'month', 'quarter', 'year', 'dayofyear']]
        X_test = self.test[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        y_test = self.test[list(df)[0]]
        
        # Performing regression with 1000 predictors, limiting tree depth to 50.
        reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=100, learning_rate=0.01)
        reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
        
        test_rmse, test_mae = self.evaluate(X_test, y_test, reg)
        
        self.plot_pred(df, fname = str(fname+'_prediction.png'))
        
        
    def evaluate(self, X_test, y_test, reg):
        self.test['prediction'] = reg.predict(X_test)
        
        # Calculating RMSE using sklearn
        test_rmse = np.sqrt(mean_squared_error(y_test, self.test['prediction']))
        print(f'The RMSE value for the test set is {test_rmse:0.2f}')
        test_mae = mean_absolute_error(y_test, self.test['prediction'])
        print(f'The MAE value for the test set is {test_mae:0.2f}')
        
        return test_rmse, test_mae
    
    def plot_pred(self, df, fname):
        df = df.merge(self.test[['prediction']], how='left', left_index=True, right_index=True)
        fig, ax = plt.subplots(figsize=(10,5))
        #ax.plot(self.test['dayofyear'], self.test[list(df)[0]], label='True load')
        #ax.plot(self.test['dayofyear'], self.test['prediction'], label='Predicted Load')
        ax.plot(self.test['hour'], self.test[list(df)[0]], label='True load')
        ax.plot(self.test['hour'], self.test['prediction'], label='Predicted Load')
        #ax.set_xlabel('Date')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Load KW')
        ax.set_title('Load prediction for January 8th 2016')
        ax.legend()
        #ax = self.test['prediction'].plot(figsize=(15, 5))
        #df['prediction'].plot(ax=ax, style='.-')
        #plt.legend(['Truth data', 'Predictions'])
        # Save the figure
        fig.savefig(os.path.join('./../plots/',fname))
        
        
        
        
        
if __name__ == '__main__':
    Xgboost()