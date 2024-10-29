import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct

import torch
import gpytorch
from torch.utils.data import DataLoader, TensorDataset

import seaborn as sns

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir) 

from Scripts.datagen_central import Datagen_central

SEED = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the GP model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        #super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        super(ExactGPModel, self).__init__(train_inputs=None, train_targets=None, likelihood=likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() + gpytorch.kernels.LinearKernel()) # + gpytorch.kernels.WhiteNoise()
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class torch_GPR_combined(object):
    def __init__(self, generator = 1):
        print("....... Loading the Dataset ........")
        data = Datagen_central()
        df = data.Gen
        
        print("..........Training for Generator " , generator, "..........")
        self.train_model(df, generator)
        
    def train_model(self, df, generator):
        print("....Splitting dataset into test and train.....")
        # Splitting the training and testing data
        self.train = df.loc[df.index < '01-07-2016']
        #self.train = df
        self.test = df.loc[df.index >= '01-08-2016']
        
        print("....generating X and y for Generator ",generator,".....")
        # Defining training and testing data
        if int(generator) == 1:
            X_train = self.train[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear', 'SG2 POWER', 'SG3 POWER', 'SG4 POWER', 'SG5 POWER']]
            y_train = self.train['SG1 POWER']

            # X_test = self.test[['hour', 'dayofweek', 'month', 'quarter', 'year', 'dayofyear']]
            X_test = self.test[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear', 'SG2 POWER', 'SG3 POWER', 'SG4 POWER', 'SG5 POWER']]
            y_test = self.test['SG1 POWER']
        elif int(generator) == 2:
            X_train = self.train[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear', 'SG1 POWER', 'SG3 POWER', 'SG4 POWER', 'SG5 POWER']]
            y_train = self.train['SG2 POWER']

            # X_test = self.test[['hour', 'dayofweek', 'month', 'quarter', 'year', 'dayofyear']]
            X_test = self.test[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear', 'SG1 POWER', 'SG3 POWER', 'SG4 POWER', 'SG5 POWER']]
            y_test = self.test['SG2 POWER']
        elif int(generator) == 3:
            X_train = self.train[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear', 'SG1 POWER', 'SG2 POWER', 'SG4 POWER', 'SG5 POWER']]
            y_train = self.train['SG3 POWER']

            # X_test = self.test[['hour', 'dayofweek', 'month', 'quarter', 'year', 'dayofyear']]
            X_test = self.test[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear', 'SG1 POWER', 'SG2 POWER', 'SG4 POWER', 'SG5 POWER']]
            y_test = self.test['SG3 POWER']
        elif int(generator) == 4:
            X_train = self.train[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear', 'SG1 POWER', 'SG2 POWER', 'SG3 POWER', 'SG5 POWER']]
            y_train = self.train['SG4 POWER']

            # X_test = self.test[['hour', 'dayofweek', 'month', 'quarter', 'year', 'dayofyear']]
            X_test = self.test[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear', 'SG1 POWER', 'SG2 POWER', 'SG3 POWER', 'SG5 POWER']]
            y_test = self.test['SG4 POWER']
        elif int(generator) == 5:
            X_train = self.train[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear', 'SG1 POWER', 'SG2 POWER', 'SG3 POWER', 'SG4 POWER']]
            y_train = self.train['SG5 POWER']

            # X_test = self.test[['hour', 'dayofweek', 'month', 'quarter', 'year', 'dayofyear']]
            X_test = self.test[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear', 'SG1 POWER', 'SG2 POWER', 'SG3 POWER', 'SG4 POWER']]
            y_test = self.test['SG5 POWER']
        
        # Replace zeros with a small constant
        X_train[X_train == 0] = 1e-3
        X_test[X_test == 0] = 1e-3

        # Add a small amount of noise to features
        noise_train = np.random.normal(0, 1e-6, X_train.shape)
        noise_test = np.random.normal(0, 1e-6, X_test.shape)
        X_train += noise_train
        X_test += noise_test
        
        # Scale the features using StandardScaler
        scaler = StandardScaler()
        print("....Scaling the train and test features.....")
        X_train_scaled = scaler.fit_transform(X_train.to_numpy())
        X_test_scaled = scaler.transform(X_test.to_numpy())
        
        print("....Creating tensors.....")
        # Convert to torch tensors
        X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_torch = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
        X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_torch = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
        
        print("....Creating batches of dataloader.....")
        # Create DataLoader for training data
        train_dataset = TensorDataset(X_train_scaled, y_train_torch)
        batch_size = 1024  # Adjust batch size as necessary
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        
        print("....moving data to device = ", device, ".....")
        # Move data to GPU if available
        #X_train_scaled = X_train_scaled.to(device)
        #y_train_torch = y_train_torch.to(device)
        #X_test_scaled = X_test_scaled.to(device)
        #y_test_torch = y_test_torch.to(device)
        
        
        print("....initializing the likelihood and the model.....")
        # Initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(X_train_scaled[:batch_size], y_train_torch[:batch_size], likelihood)
        
        
        print("....moving the model and likelihood to device = ", device, ".....")
        # Move model to GPU if available
        model = model.to(device)
        likelihood = likelihood.to(device)
        
        
        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        
        # Use the Adam optimizer
        optimizer = torch.optim.Adam([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=2.0)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        training_iterations = 280
        
        Loss = []
        
        for epoch in range(training_iterations):
            loss_epoch = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                # Set the current batch as the training data
                model.set_train_data(inputs=X_batch, targets=y_batch, strict=False)
                # Add jitter to ensure numerical stability
                with gpytorch.settings.cholesky_jitter(1e-1):
                    output = model(X_batch)
                    loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                loss_epoch += loss
            Loss.append(loss_epoch.cpu().detach().numpy())
            print(f'Iteration {epoch+1}/{training_iterations} - Loss: {loss_epoch.item()}')
        
        '''
        for i in range(training_iterations):
            optimizer.zero_grad()
            output = model(X_train_scaled)
            loss = -mll(output, y_train_torch)
            loss.backward()
            print(f'Iteration {i+1}/{training_iterations} - Loss: {loss.item()}')
            optimizer.step()
        '''
        print("Evaluating the GPR Model for Generator" + str(generator))
        # Switch to evaluation mode
        model.eval()
        likelihood.eval()
        
        print("Sequentially predicting on and retuning the GPR Model for Generator" + str(generator))
        self.sequential_predict(model, likelihood, X_train_scaled, y_train_torch, X_test_scaled, y_test_torch, batch_size = batch_size)
        print("Plotting the prediction on the GPR Model for Generator" + str(generator))
        self.plot_pred(df, fname = str('Generator_'+str(generator) +'_gpr_prediction_combined_known_init.png'), generator=generator)
        print("Plotting the training loss of the GPR Model for Generator" + str(generator))
        self.plot_loss(iterations = training_iterations, loss=Loss, fname = str('Generator_'+str(generator) +'_gpr_train_loss_combined_known_init.png'))
    
    # Function to sequentially predict using the initial value
    def sequential_predict(self, model, likelihood, X_train_scaled, y_train_torch, X_test_scaled, y_test_torch, batch_size = 64):
        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        total_loss = 0  # Initialize total loss
        
        # Make predictions with the initial value consideration
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_preconditioner_size(10), gpytorch.settings.cholesky_jitter(1e-1):
            initial_y_value = y_test_torch[0].to(device)
            predictions = [initial_y_value.item()]

            for i in range(1, len(X_test_scaled)):
                current_X_train = torch.cat((X_train_scaled, X_test_scaled[:i]), dim=0)
                current_y_train = torch.cat((y_train_torch, torch.tensor(predictions[:i])), dim=0) #, device=device
                
                current_dataset = TensorDataset(current_X_train, current_y_train)
                #batch_size = 64  # Adjust batch size as necessary
                current_loader = DataLoader(current_dataset, batch_size=batch_size, shuffle=False)

                # Retrain model with the new data
                for X_batch, y_batch in current_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)
                    model.set_train_data(inputs=X_batch, targets=y_batch, strict=False)
                    # Calculate loss for current prediction
                    #loss = -mll(model(X_batch), y_batch)
                    #total_loss += loss.item()

                # Make prediction
                with gpytorch.settings.fast_pred_var():
                    pred = likelihood(model(X_test_scaled[i].unsqueeze(0).to(device))).mean
                    predictions.append(pred.item()) #.cpu()
        
        
        #with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.max_preconditioner_size(10), gpytorch.settings.cholesky_jitter(1e-1):
        #    predictions = likelihood(model(X_test_scaled.unsqueeze(0).to(device))).mean
        # Print total loss
        print("Total loss:", total_loss)
        self.test['prediction'] = predictions

    
    def plot_pred(self, df, fname, generator=1):
        df = df.merge(self.test[['prediction']], how='left', left_index=True, right_index=True)
        fig, ax = plt.subplots(figsize=(10,5))
        #ax.plot(self.test['dayofyear'], self.test[list(df)[0]], label='True load')
        #ax.plot(self.test['dayofyear'], self.test['prediction'], label='Predicted Load')
        if int(generator) == 1:
            ax.plot(self.test['hour'], self.test['SG1 POWER'], label='True load')
        elif int(generator) == 2:
            ax.plot(self.test['hour'], self.test['SG2 POWER'], label='True load')
        elif int(generator) == 3:
            ax.plot(self.test['hour'], self.test['SG3 POWER'], label='True load')
        elif int(generator) == 4:
            ax.plot(self.test['hour'], self.test['SG4 POWER'], label='True load')
        elif int(generator) == 5:
            ax.plot(self.test['hour'], self.test['SG5 POWER'], label='True load')
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
        
    def plot_loss(self, iterations, loss, fname):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(loss, label='Training Loss') #np.array(iterations+1), 
        ax.set_xlabel('Number of Iterations')
        ax.set_ylabel('marginal log likelihood loss')
        ax.set_title('Training loss')
        ax.legend()
        fig.savefig(os.path.join('./../plots/',fname))
if __name__ == '__main__':
    torch_GPR_combined(sys.argv[1])