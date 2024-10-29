import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datagen import Datagen

clr_palet = sns.color_palette()
plt.style.use("dark_background")


class Data_visualize(object):
    def __init__(self):
        data = Datagen()
        
        df1 = data.Gen1
        self.plot_monthly(df1, './../plots/Gen1_monthly.png')
        self.plot_hourly(df1, './../plots/Gen1_hourly.png')
        
        df2 = data.Gen2
        self.plot_monthly(df2, './../plots/Gen2_monthly.png')
        self.plot_hourly(df2, './../plots/Gen2_hourly.png')
        
        df3 = data.Gen3
        self.plot_monthly(df3, './../plots/Gen3_monthly.png')
        self.plot_hourly(df3, './../plots/Gen3_hourly.png')
        
        df4 = data.Gen4
        self.plot_monthly(df4, './../plots/Gen4_monthly.png')
        self.plot_hourly(df4, './../plots/Gen4_hourly.png')
        
        df5 = data.Gen5
        self.plot_monthly(df5, './../plots/Gen5_monthly.png')
        self.plot_hourly(df5, './../plots/Gen5_hourly.png')
        
    def plot_monthly(self, df, fname):
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(data=df, x='month', y=list(df)[0], palette='Blues')
        ax.set_title('MW by month')
        fig.savefig(fname)
        
        
    def plot_hourly(self, df, fname):
        fig, ax = plt.subplots(figsize=(10,6))
        sns.boxplot(data=df, x='hour', y=list(df)[0], palette='Blues')
        ax.set_title('MW by month')
        fig.savefig(fname)
        
        
if __name__ == '__main__':
    Data_visualize()
        
        