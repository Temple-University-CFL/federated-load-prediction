import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn

from tqdm import tqdm

import numpy as np

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir) 

from Scripts.datagen import Datagen

SEED = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

class utils(object):
    def __init__(self, generator=1):
        self.trainloader, self.testloader = self.load_data(generator = generator)
        
        
        
    def load_data(self, generator=1):
        # Define batch size
        batch_size = 64
        data = Datagen()
        if int(generator) == 1:
            df = data.Gen1
            print(".....Loading Genertator 1........")
        elif int(generator) == 2:
            df = data.Gen2
            print(".....Loading Genertator 2........")
        elif int(generator) == 3:
            df = data.Gen3
            print(".....Loading Genertator 3........")
        elif int(generator) == 4:
            df = data.Gen4
            print(".....Loading Genertator 4........")
        elif int(generator) == 5:
            df = data.Gen5
            print(".....Loading Genertator 5........")
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
        X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=SEED)        
        # Create train Dataset
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        # Create train DataLoader
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)        
        # Create test Dataset
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        # Create test DataLoader
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        return trainloader, testloader
    
    def train(self, net, trainloader, epochs):
        """Train the model on the training set."""
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.5)
        for _ in range(epochs):
            normalized_mse, normalized_mae, Loss = 0.0, 0.0, 0.0
            for batch in tqdm(trainloader, "Training"):
                X_batch, y_batch = batch
                # Move tensors to CUDA if available
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                X_batch = X_batch.unsqueeze(1)
                # Calculate variance of target variable y
                y_batch_var = np.var(y_batch.cpu().numpy())
                net.train()
                optimizer.zero_grad()
                outputs = net(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                #Loss.append(loss.item())
                Loss += loss
                loss.backward(retain_graph=True)
                optimizer.step()
                with torch.no_grad():
                    predictions = net(X_batch)
                    mse = mean_squared_error(y_batch.numpy(), predictions.squeeze().cpu().numpy())
                    # Calculate normalized MSE for training set
                    normalized_mse += mse / y_batch_var
                    mae = mean_absolute_error(y_batch.numpy(), predictions.squeeze().cpu().numpy())
                    normalized_mse += mae / y_batch_var
                    #MSE.append(normalized_mse)
                    #MAE.append(normalized_mae)
                    
                    
                    
    def test(self, net, testloader):
        """Validate the model on the test set."""
        criterion = nn.MSELoss()
        # Move model to CUDA if available
        net.to(device)
        # Evaluate the model
        net.eval()
        
        normalized_mse, normalized_mae, loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(testloader, "Testing"):
                X_batch, y_batch = batch
                # Move tensors to CUDA if available
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                X_batch = X_batch.unsqueeze(1)
                test_predictions = net(X_batch)
                loss += criterion(test_predictions.squeeze(), y_batch).item()
                test_mse = mean_squared_error(y_batch.numpy(), test_predictions.squeeze().cpu().numpy())
                test_mae = mean_absolute_error(y_batch.numpy(), test_predictions.squeeze().cpu().numpy())
                # Calculate variance of target variable y
                y_batch_var = np.var(y_batch.cpu().numpy())
                normalized_mse += test_mse / y_batch_var
                normalized_mae += test_mae / y_batch_var
        
        return loss, normalized_mse
    
    
if __name__ == "__main__":
    utils(sys.argv[1])
