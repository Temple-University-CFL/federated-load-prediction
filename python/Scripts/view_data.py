import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline

from load_data import Load_data

clr_palet = sns.color_palette()
plt.style.use("dark_background")

class View_data(object):
    def __init__(self):
        data = Load_data()
        
        df1 = data.Z1
        grouped1 = self.group_data(df1)
        # Plot each group
        for (dayofyear, year), group in grouped1:
            # Fit a polynomial to the data
            poly_params = np.polyfit(group['hour'], group['Z1 (W)'], 10)
            poly_model = np.poly1d(poly_params)

            # Create smooth data points
            x_smooth = np.linspace(group['hour'].min(), group['hour'].max(), 300)
            y_smooth = poly_model(x_smooth)
            # Plot the smoothed curve
            plt.plot(x_smooth, y_smooth, label=f'Day {dayofyear} - Year {year}')
            #plt.plot(group['hour'], group['Z1 (W)'], label=f'Day {dayofyear} - Year {year}')
        # Add labels and legend
        plt.xlabel('Hour of Day')
        plt.ylabel('Z1')
        plt.title('Z1 over 24 hours for different days of the year across different years')
        #plt.legend()
        plt.grid(True)
        plt.savefig("./../plots/Z1.png")
        
        df2 = data.Z2_2s
        grouped2 = self.group_data(df2)
        # Plot each group
        for (dayofyear, year), group in grouped2:
            # Fit a polynomial to the data
            poly_params = np.polyfit(group['hour'], group['Z2 (W)'], 10)
            poly_model = np.poly1d(poly_params)

            # Create smooth data points
            x_smooth = np.linspace(group['hour'].min(), group['hour'].max(), 300)
            y_smooth = poly_model(x_smooth)
            # Plot the smoothed curve
            plt.plot(x_smooth, y_smooth, label=f'Day {dayofyear} - Year {year}')
            #plt.plot(group['hour'], group['Z1 (W)'], label=f'Day {dayofyear} - Year {year}')
        # Add labels and legend
        plt.xlabel('Hour of Day')
        plt.ylabel('Z2')
        plt.title('Z2 over 24 hours for different days of the year across different years')
        #plt.legend()
        plt.grid(True)
        plt.savefig("./../plots/Z2.png")
        
        # Plot each group
        for (dayofyear, year), group in grouped2:
            # Fit a polynomial to the data
            poly_params = np.polyfit(group['hour'], group['2S (W)'], 10)
            poly_model = np.poly1d(poly_params)

            # Create smooth data points
            x_smooth = np.linspace(group['hour'].min(), group['hour'].max(), 300)
            y_smooth = poly_model(x_smooth)
            # Plot the smoothed curve
            plt.plot(x_smooth, y_smooth, label=f'Day {dayofyear} - Year {year}')
            #plt.plot(group['hour'], group['Z1 (W)'], label=f'Day {dayofyear} - Year {year}')
        # Add labels and legend
        plt.xlabel('Hour of Day')
        plt.ylabel('2S (W)')
        plt.title('2S over 24 hours for different days of the year across different years')
        #plt.legend()
        plt.grid(True)
        plt.savefig("./../plots/2S.png")
        
        df3 = data.Z3
        grouped3 = self.group_data(df3)
        
        # Plot each group
        for (dayofyear, year), group in grouped3:
            # Fit a polynomial to the data
            poly_params = np.polyfit(group['hour'], group['Z3 (W)'], 10)
            poly_model = np.poly1d(poly_params)

            # Create smooth data points
            x_smooth = np.linspace(group['hour'].min(), group['hour'].max(), 300)
            y_smooth = poly_model(x_smooth)
            # Plot the smoothed curve
            plt.plot(x_smooth, y_smooth, label=f'Day {dayofyear} - Year {year}')
            #plt.plot(group['hour'], group['Z1 (W)'], label=f'Day {dayofyear} - Year {year}')
        # Add labels and legend
        plt.xlabel('Hour of Day')
        plt.ylabel('Z3 (W)')
        plt.title('Z3 over 24 hours for different days of the year across different years')
        #plt.legend()
        plt.grid(True)
        plt.savefig("./../plots/Z3.png")
        
        df4 = data.Z4_3s
        grouped4 = self.group_data(df4)
        
        # Plot each group
        for (dayofyear, year), group in grouped4:
            # Fit a polynomial to the data
            poly_params = np.polyfit(group['hour'], group['Z4 (W)'], 10)
            poly_model = np.poly1d(poly_params)

            # Create smooth data points
            x_smooth = np.linspace(group['hour'].min(), group['hour'].max(), 300)
            y_smooth = poly_model(x_smooth)
            # Plot the smoothed curve
            plt.plot(x_smooth, y_smooth, label=f'Day {dayofyear} - Year {year}')
            #plt.plot(group['hour'], group['Z1 (W)'], label=f'Day {dayofyear} - Year {year}')
        # Add labels and legend
        plt.xlabel('Hour of Day')
        plt.ylabel('Z4 (W)')
        plt.title('Z4 over 24 hours for different days of the year across different years')
        #plt.legend()
        plt.grid(True)
        plt.savefig("./../plots/Z4.png")
        
        # Plot each group
        for (dayofyear, year), group in grouped4:
            # Fit a polynomial to the data
            poly_params = np.polyfit(group['hour'], group['Z4-3S (W)'], 10)
            poly_model = np.poly1d(poly_params)

            # Create smooth data points
            x_smooth = np.linspace(group['hour'].min(), group['hour'].max(), 300)
            y_smooth = poly_model(x_smooth)
            # Plot the smoothed curve
            plt.plot(x_smooth, y_smooth, label=f'Day {dayofyear} - Year {year}')
            #plt.plot(group['hour'], group['Z1 (W)'], label=f'Day {dayofyear} - Year {year}')
        # Add labels and legend
        plt.xlabel('Hour of Day')
        plt.ylabel('3S (W)')
        plt.title('3S over 24 hours for different days of the year across different years')
        #plt.legend()
        plt.grid(True)
        plt.savefig("./../plots/3S.png")
        
        df5 = data.Z5_5s
        grouped5 = self.group_data(df5)
        
        # Plot each group
        for (dayofyear, year), group in grouped5:
            # Fit a polynomial to the data
            poly_params = np.polyfit(group['hour'], group['Z5 (W)'], 10)
            poly_model = np.poly1d(poly_params)

            # Create smooth data points
            x_smooth = np.linspace(group['hour'].min(), group['hour'].max(), 300)
            y_smooth = poly_model(x_smooth)
            # Plot the smoothed curve
            plt.plot(x_smooth, y_smooth, label=f'Day {dayofyear} - Year {year}')
            #plt.plot(group['hour'], group['Z1 (W)'], label=f'Day {dayofyear} - Year {year}')
        # Add labels and legend
        plt.xlabel('Hour of Day')
        plt.ylabel('Z5 (W)')
        plt.title('Z5 over 24 hours for different days of the year across different years')
        #plt.legend()
        plt.grid(True)
        plt.savefig("./../plots/Z5.png")
        
        # Plot each group
        for (dayofyear, year), group in grouped5:
            # Fit a polynomial to the data
            poly_params = np.polyfit(group['hour'], group['Z5-5S'], 10)
            poly_model = np.poly1d(poly_params)

            # Create smooth data points
            x_smooth = np.linspace(group['hour'].min(), group['hour'].max(), 300)
            y_smooth = poly_model(x_smooth)
            # Plot the smoothed curve
            plt.plot(x_smooth, y_smooth, label=f'Day {dayofyear} - Year {year}')
            #plt.plot(group['hour'], group['Z1 (W)'], label=f'Day {dayofyear} - Year {year}')
        # Add labels and legend
        plt.xlabel('Hour of Day')
        plt.ylabel('5S (W)')
        plt.title('5S over 24 hours for different days of the year across different years')
        #plt.legend()
        plt.grid(True)
        plt.savefig("./../plots/5S.png")
        
        df6 = data.Z6
        grouped6 = self.group_data(df6)
        
        # Plot each group
        for (dayofyear, year), group in grouped6:
            # Fit a polynomial to the data
            poly_params = np.polyfit(group['hour'], group['Z6 (W)'], 10)
            poly_model = np.poly1d(poly_params)

            # Create smooth data points
            x_smooth = np.linspace(group['hour'].min(), group['hour'].max(), 300)
            y_smooth = poly_model(x_smooth)
            # Plot the smoothed curve
            plt.plot(x_smooth, y_smooth, label=f'Day {dayofyear} - Year {year}')
            #plt.plot(group['hour'], group['Z1 (W)'], label=f'Day {dayofyear} - Year {year}')
        # Add labels and legend
        plt.xlabel('Hour of Day')
        plt.ylabel('Z6 (W)')
        plt.title('Z6 over 24 hours for different days of the year across different years')
        #plt.legend()
        plt.grid(True)
        plt.savefig("./../plots/Z6.png")
        
        df7 = data.Z7
        grouped7 = self.group_data(df7)
        
        # Plot each group
        for (dayofyear, year), group in grouped7:
            # Fit a polynomial to the data
            poly_params = np.polyfit(group['hour'], group['Z7 (W)'], 10)
            poly_model = np.poly1d(poly_params)

            # Create smooth data points
            x_smooth = np.linspace(group['hour'].min(), group['hour'].max(), 300)
            y_smooth = poly_model(x_smooth)
            # Plot the smoothed curve
            plt.plot(x_smooth, y_smooth, label=f'Day {dayofyear} - Year {year}')
            #plt.plot(group['hour'], group['Z1 (W)'], label=f'Day {dayofyear} - Year {year}')
        # Add labels and legend
        plt.xlabel('Hour of Day')
        plt.ylabel('Z7 (W)')
        plt.title('Z7 over 24 hours for different days of the year across different years')
        #plt.legend()
        plt.grid(True)
        plt.savefig("./../plots/Z7.png")
        
        df8 = data.Z8
        grouped8 = self.group_data(df8)
        
        # Plot each group
        for (dayofyear, year), group in grouped8:
            # Fit a polynomial to the data
            poly_params = np.polyfit(group['hour'], group['Z8 (W)'], 10)
            poly_model = np.poly1d(poly_params)

            # Create smooth data points
            x_smooth = np.linspace(group['hour'].min(), group['hour'].max(), 300)
            y_smooth = poly_model(x_smooth)
            # Plot the smoothed curve
            plt.plot(x_smooth, y_smooth, label=f'Day {dayofyear} - Year {year}')
            #plt.plot(group['hour'], group['Z1 (W)'], label=f'Day {dayofyear} - Year {year}')
        # Add labels and legend
        plt.xlabel('Hour of Day')
        plt.ylabel('Z8 (W)')
        plt.title('Z8 over 24 hours for different days of the year across different years')
        #plt.legend()
        plt.grid(True)
        plt.savefig("./../plots/Z8.png")
        
    def group_data(self, df):
        # Group data by 'dayofyear' and 'year'
        grouped = df.groupby(['dayofyear', 'year'])
        return grouped
    
if __name__ == '__main__':
    View_data()