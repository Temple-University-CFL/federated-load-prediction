import pandas as pd
import glob 
# Date wrangling
from datetime import datetime, timedelta
import os
import sys

class Load_data(object):
    def __init__(self, filepath = './../../load_data'):
        csv_files = glob.glob(os.path.join(filepath, "*.xlsx"))
        i=0
        for f in csv_files:
            if i==0:
                self.Z1 = self.read_data(f)
                i+=1
            elif i == 1:
                #self.Z2_2s = self.read_data(f)
                self.Z6 = self.read_data(f)
                i+=1
            elif i == 2:
                #self.Z3 = self.read_data(f)
                self.Z7 = self.read_data(f)
                i+=1
            elif i == 3:
                #self.Z4_3s = self.read_data(f)
                self.Z5_5s = self.read_data(f)
                i+=1
            elif i == 4:
                #self.Z5_5s = self.read_data(f)
                self.Z4_3s = self.read_data(f)
                i+=1
            elif i == 5:
                #self.Z6 = self.read_data(f)
                self.Z2_2s = self.read_data(f)
                i+=1
            elif i == 6:
                #self.Z7 = self.read_data(f)
                self.Z8 = self.read_data(f)
                i+=1
            elif i == 7:
                #self.Z8 = self.read_data(f)
                self.Z3 = self.read_data(f)
                i+=1
            
        
        
        
    def read_data(self, f):
        format_data = "%Y-%m-%d %H:%M:%S"
        # Read specific sheets
        df = pd.read_excel(f, sheet_name='Sheet1')
        # Combine the Date and Time columns into a single column
        df['DateTime'] = df['Date'].astype(str) + ' ' + df['Time'].astype(str)
        #    print(datetime.strptime(str(x), format_data))
        df['DateTime'] = [datetime.strptime(str(x), format_data) for x in df['DateTime']]
        
        #df['DateTime'] = pd.to_datetime(df['DateTime'])

        df = df.set_index("DateTime")
        # Creating features for personalized predictions
        df = self.addfeat(df)
        #print(f"Sheet name: {sheet_name}")
        #print(df)
        # Print the header of the DataFrame
        #print(df.head())
        # Convert 'DateTime' columns to datetime format
        

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
    Load_data()