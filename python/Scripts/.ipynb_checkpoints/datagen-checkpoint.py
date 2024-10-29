import pandas as pd
# Date wrangling
from datetime import datetime, timedelta
import os


class Datagen(object):
    def __init__(self, filepath = './../../dataset'):
        sheet_names = ['Sheet1', 'Sheet2', 'Sheet3', 'Sheet4', 'Sheet5']
        
        self.Gen1 = self.read_data(filepath, 'Sheet1')
        #print(self.Gen1)
        self.Gen2 = self.read_data(filepath, 'Sheet2')
        self.Gen3 = self.read_data(filepath, 'Sheet3')
        self.Gen4 = self.read_data(filepath, 'Sheet4')
        self.Gen5 = self.read_data(filepath, 'Sheet5')
        
        #print(list(self.Gen1)[0])
    
    def read_data(self, filepath, sheet_name):
        format_data = "%Y-%m-%d %H:%M:%S"
        csvfile = os.path.join(filepath,'Ship1.xlsx')
        # Read specific sheets
        df = pd.read_excel(csvfile, sheet_name=sheet_name)
        #for x in df['DateTime']:
        #    print(datetime.strptime(str(x), format_data))
        df['DateTime'] = [datetime.strptime(str(x), format_data) for x in df['DateTime']]
        
        df = df.set_index("DateTime")
        # Creating features for personalized predictions
        df = self.addfeat(df)
        #print(f"Sheet name: {sheet_name}")
        #print(df)

        return df


    def addfeat(self, df):

        df = df.copy()
        df['minute'] = df.index.minute
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.day_of_week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        df['dayofyear'] = df.index.day_of_year
        return df
    
if __name__ == '__main__':
    Datagen()