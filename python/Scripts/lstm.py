import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from Scripts.datagen import Datagen


SEED = 42
# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
    
class main(object):
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
        num_epochs = 100
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        
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
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_torch = torch.tensor(y_train.values, dtype=torch.float32)
        
        # Convert to PyTorch tensors
        X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_torch = torch.tensor(y_test.values, dtype=torch.float32)
        
        '''
        # Split features and target
        X = df[['hour', 'dayofweek', 'month', 'quarter', 'year', 'dayofyear']]
        y = df[list(df)[0]]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32)
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, shuffle=False, random_state=SEED)
        '''

        
        #print("training start")
        model, train_loss, train_RMSE, train_MAE = self.trainx(X_train_torch, X_test_torch, y_train_torch, y_test_torch, criterion = criterion, num_epochs = num_epochs)
        test_loss, test_rmse, test_mae, test_predictions = self.evaluate(model, criterion, X_test_torch, y_test_torch)
        
        self.plot_loss(loss = train_loss, fname = str(fname+'_Loss.png'), num_epochs = num_epochs)
        self.plot_error(mse = train_RMSE, mae = train_MAE, fname= str(fname+'_Error.png'), num_epochs = num_epochs)
        # self.plot_pred(X_test = X_test, y_test = y_test, test_predictions = test_predictions, fname = str(fname+'_lstm_prediction.png'))
        self.plot_pred(df, fname = str(fname+'_lstm_prediction.png'))
        
    def trainx(self, X_train, X_test, y_train, y_test, criterion = nn.MSELoss(), num_epochs = 100):
        #print("train start")
        Loss = []
        MSE = []
        MAE = []
        #criterion = nn.MSELoss()
        #num_epochs = 100
        # Initialize the model
        input_size = X_train.shape[1]
        hidden_size = 64
        output_size = 1
        model = LSTMModel(input_size, hidden_size, output_size)
        
        # Move model to CUDA if available
        model.to(device)

        # Move tensors to CUDA if available
        X_train = X_train.to(device)
        y_train = y_train.to(device)

        # Define loss function and optimizer
        criterion = criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
        
        X_train = X_train.unsqueeze(1)
        
        # Calculate variance of target variable y
        y_train_var = np.var(y_train.cpu().numpy())
        
        # Training loop
        num_epochs = num_epochs
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            Loss.append(loss.item())
            loss.backward()
            optimizer.step()
                
            with torch.no_grad():
                predictions = model(X_train)
                mse = mean_squared_error(y_train.numpy(), predictions.squeeze().cpu().numpy())
                # Calculate normalized MSE for training set
                normalized_mse = mse / y_train_var
                mae = mean_absolute_error(y_train.numpy(), predictions.squeeze().cpu().numpy())
                normalized_mae = mae / y_train_var
                #MSE.append(normalized_mse)
                MSE.append(mse)
                MAE.append(normalized_mae)
                if (epoch+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, MSE: {mse:.4f}, , MAE: {mae:.4f}')
                
        return model, Loss, np.sqrt(MSE), MAE
    
    def evaluate(self, model, criterion, X_test, y_test):
        X_test = X_test.unsqueeze(1)
        # Move model to CUDA if available
        model.to(device)
        # Move tensors to CUDA if available
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        # Evaluate the model
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions.squeeze(), y_test)
            print(f'Test Loss: {test_loss.item():.4f}')
            test_rmse = np.sqrt(mean_squared_error(y_test.numpy(), test_predictions.squeeze().cpu().numpy()))
            test_mae = mean_absolute_error(y_test.numpy(), test_predictions.squeeze().cpu().numpy())
            print(f'Testing RMSE: {test_rmse:.4f}')
            print(f'Testing MAE: {test_mae:.4f}')
        self.test['prediction'] = test_predictions
        
        return test_loss, test_rmse, test_mae, test_predictions
    
    def plot_loss(self, loss, fname, num_epochs = 100):
        # Plot training loss
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(range(1, num_epochs+1), loss, label='Training Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        # Save the figure
        fig.savefig(os.path.join('./../plots/',fname))
        
    def plot_error(self, mse, mae, fname, num_epochs = 100):
        # Plot training error
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(range(1, num_epochs+1), mse, label='RMSE')
        ax.plot(range(1, num_epochs+1), mae, label='MAE')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Error')
        ax.set_title('Training Error')
        ax.legend()
        # Save the figure
        fig.savefig(os.path.join('./../plots/',fname))
    
    '''
    def plot_pred(self, X_test, y_test, test_predictions, fname):
        # Plot training error
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(y_test, label='True load')
        ax.plot(test_predictions, label='Predicted load')
        ax.set_xlabel('data points')
        ax.set_ylabel('Load')
        ax.set_title('Load prediction')
        ax.legend()
        # Save the figure
        fig.savefig(os.path.join('./../plots/',fname))
        
    '''
        
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
    main()

        