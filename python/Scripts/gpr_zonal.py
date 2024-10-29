import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, DotProduct

import torch
import torch.nn as nn
import gpytorch
from torch.utils.data import DataLoader, TensorDataset

import seaborn as sns

import asyncio
import threading

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir) 

from Scripts.load_data import Load_data

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
    
# Define a polynomial function
def polynomial_function(x):
    full_load = 2500*(10**3)
    p1d = 2.314*(10**(-6))
    p2d = -3.295*(10**(-4))
    p3d = -3.99*(10**(-3))
    p4d = 2.537
    p5d = 19.21
    x = (x/full_load)*100
    # Calculate the fuel efficiency for each vector in x
    fuel_eff = (p1d * (x**4) + 
                p2d * (x**3) + 
                p3d * (x**2) + 
                p4d * x + 
                p5d * torch.ones_like(x))
    #return torch.stack([x**2, x**3, x**4], dim=-1)
    #return torch.stack([p1d*(x**4) + p2d*(x**3) + p3d*(x**2) + p4d*(x) + p5d], dim=-1)
    return fuel_eff

# Define a function to load GP layers from saved files
def load_gp_layers(layer, filenames, train_x, train_y):
    for i, filename in enumerate(filenames):
        state_dict = torch.load(filename)
        layer[i].load_state_dict(state_dict)
        layer[i].likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Define the Bayesian neural network
class BayesianNN(nn.Module):
    def __init__(self, train_x, train_y, gp_layers_Gm, gp_layers_Zm):
        super(BayesianNN, self).__init__()
        self.gp_layers_Gm = nn.ModuleList(gp_layers_Gm)
        self.gp_layers_Zm = nn.ModuleList(gp_layers_Zm)
        load_gp_layers(self.gp_layers_Gm, gm_filenames, train_x, train_y)
        load_gp_layers(self.gp_layers_Zm, zm_filenames, train_x, train_y)
        self.output_layer = GPRegressionLayer(
            torch.cat([train_x] * 14, dim=-1),  # 3 Gm + 8 Zm + 3 f(x)
            train_y,  # Placeholder, we'll fit this separately
            gpytorch.likelihoods.GaussianLikelihood()
        )

    def forward(self, x):
        G_outputs = [layer(x).mean.unsqueeze(-1) for layer in self.gp_layers_Gm]
        Z_outputs = [layer(x).mean.unsqueeze(-1) for layer in self.gp_layers_Zm]
        f_outputs = polynomial_function(torch.cat(G_outputs, dim=-1))
        all_outputs = torch.cat(G_outputs + Z_outputs + [f_outputs], dim=-1)
        return self.output_layer(all_outputs)
    
    
class torch_GPR_zonal(object):
    def __init__(self, rounds = 8):
        self.batch_size = 1024
        data = Load_data()
        self.df1 = data.Z1
        #self.df1.set_index('DateTime', inplace=True)
        self.df2 = data.Z2_2s
        #self.df2.set_index('DateTime', inplace=True)
        self.df3 = data.Z3
        #self.df3.set_index('DateTime', inplace=True)
        self.df4 = data.Z4_3s
        #self.df4.set_index('DateTime', inplace=True)
        self.df5 = data.Z5_5s
        #self.df5.set_index('DateTime', inplace=True)
        self.df6 = data.Z6
        #self.df6.set_index('DateTime', inplace=True)
        self.df7 = data.Z7
        #self.df7.set_index('DateTime', inplace=True)
        self.df8 = data.Z8
        #self.df8.set_index('DateTime', inplace=True)
        
        # train and load the zoanl model in an array 
        #self.gp_layers_Zm = self.load_zonal_models()
        self.load_zonal_models()
        # train and load initialized generator model
        #self.gp_layers_Gm = self.loaf_gen_init_model()
        self.loaf_gen_init_model()
        #self.merged_df = self.sync_data()
        
        # init loss
        self.Loss_2s = []
        self.Loss_3s = []
        self.Loss_5s = []
        generator_values = ['2S', '3S','5S']
        for i in range(int(rounds)):
            self.start_thread(generator_values)
        
    def start_thread(self, generator_values):
        # List to keep track of threads
        threads = []
        # Create and start a thread for each generator value
        for generator in generator_values:
            thread = threading.Thread(target=self.train_generator, args=(generator))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
    def sync_data(self):
        # Resample both DataFrames to a common frequency (e.g., 1 minute)
        df1_resampled = self.df1.resample('min').mean()
        df2_resampled = self.df2.resample('min').mean()
        df3_resampled = self.df3.resample('min').mean()
        df4_resampled = self.df4.resample('min').mean()
        df5_resampled = self.df5.resample('min').mean()
        df6_resampled = self.df6.resample('min').mean()
        df7_resampled = self.df7.resample('min').mean()
        df8_resampled = self.df8.resample('min').mean()
        
        # Merge the DataFrames on the resampled 'DateTime' index
        merged_df = pd.merge(df1_resampled, df2_resampled, df3_resampled, df4_resampled, df5_resampled, df6_resampled, df7_resampled, df8_resampled, left_index=True, right_index=True, how='outer')

        # Handle missing values (optional, here using forward fill)
        merged_df.fillna(method='ffill', inplace=True)
        
        # Print the merged DataFrame
        print(merged_df.head())
        return merged_df
    
    def zonal_model(self, zone): 
        if zone == "Z1":
            Y = self.df1['Z1 (W)']
            X = self.df1[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        elif zone == "Z2":
            Y = self.df2['Z2 (W)']
            X = self.df2[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        elif zone == "Z3":
            Y = self.df3['Z3 (W)']
            X = self.df3[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        elif zone == "Z4":
            Y = self.df4['Z4 (W)']
            X = self.df4[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        elif zone == "Z5":
            Y = self.df5['Z5 (W)']
            X = self.df5[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        elif zone == "Z6":
            Y = self.df6['Z6 (W)']
            X = self.df6[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        elif zone == "Z7":
            Y = self.df7['Z7 (W)']
            X = self.df7[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        elif zone == "Z8":
            Y = self.df8['Z8 (W)']
            X = self.df8[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        else:
            raise ValueError("Unknown Zone")
        
        # Scale the features using StandardScaler
        scaler = StandardScaler()
        print("....Scaling the zonal features.....")
        X_train_scaled = scaler.fit_transform(X.to_numpy())
        
        print("....Creating tensors.....")
        # Convert to torch tensors
        X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_torch = torch.tensor(Y.to_numpy(), dtype=torch.float32)
        
        print("....Creating batches of dataloader.....")
        # Create DataLoader for training data
        train_dataset = TensorDataset(X_train_scaled, y_train_torch)
        batch_size = self.batch_size  # Adjust batch size as necessary
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        print("....initializing the likelihood and the zonal model.....")
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
        ], lr=0.2)
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
        return model
    
    def init_gen_model(self, gen):
        if gen == "2S":
            Y = self.df2['2S (W)']
            X = self.df2[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        elif gen == "3S":
            Y = self.df4['Z4-3S (W)']
            X = self.df4[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        elif gen == "5S":
            Y = self.df5['Z5-5S']
            X = self.df5[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
        else:
            raise ValueError("Unknown Generator")
            
        # Scale the features using StandardScaler
        scaler = StandardScaler()
        print("....Scaling the initial Generator features.....")
        X_train_scaled = scaler.fit_transform(X.to_numpy())
        
        print("....Creating tensors.....")
        # Convert to torch tensors
        X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_torch = torch.tensor(Y.to_numpy(), dtype=torch.float32)
        
        print("....Creating batches of dataloader.....")
        # Create DataLoader for training data
        train_dataset = TensorDataset(X_train_scaled, y_train_torch)
        batch_size = self.batch_size  # Adjust batch size as necessary
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        print("....initializing the likelihood and the initial Generator model.....")
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
        ], lr=0.2)
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
        return model
    
    def load_zonal_models(self):
        Zm1 = self.zonal_model(zone = "Z1")
        Zm2 = self.zonal_model(zone = "Z2")
        Zm3 = self.zonal_model(zone = "Z3")
        Zm4 = self.zonal_model(zone = "Z4")
        Zm5 = self.zonal_model(zone = "Z5")
        Zm6 = self.zonal_model(zone = "Z6")
        Zm7 = self.zonal_model(zone = "Z7")
        Zm8 = self.zonal_model(zone = "Z8")
        
        self.gp_layers_Zm = [Zm1,Zm2,Zm3,Zm4,Zm5,Zm6,Zm7,Zm8]
        
    def loaf_gen_init_model(self):
        gm1 = self.init_gen_model(gen == "2S")
        gm2 = self.init_gen_model(gen == "3S")
        gm3 = self.init_gen_model(gen == "5S")
        
        self.gp_layers_Gm = [gm1, gm2,gm3]
        
    def train_model(self, model, likelihood, optimizer, train_loader, num_epochs=100):
        Loss = []
        for epoch in range(num_epochs):
            loss_epoch = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = -mll(output, batch_y)
                loss.backward()
                optimizer.step()
                loss_epoch += loss
            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{num_epochs} - Loss: {loss.item()}')
            Loss.append(loss_epoch.cpu().detach().numpy())
        return model, Loss
        
    async def train_generator(self, generator):
        train_date = '07-05-2015'
        test_date = '07-06-2015'
        if generator == "2S":
            df = self.df2
            # Splitting the training and testing data
            self.train = df.loc[df.index < train_date]
            #self.train = df
            self.test = df.loc[df.index >= test_date]
            X_train = self.train[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
            y_train = self.train['2S (W)']
            X_test = self.test[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
            y_test = self.test['2S (W)']
        elif generator == "3S":
            df = self.df4
            # Splitting the training and testing data
            self.train = df.loc[df.index < train_date]
            #self.train = df
            self.test = df.loc[df.index >= test_date]
            X_train = self.train[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
            y_train = self.train['Z4-3S (W)']
            X_test = self.test[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
            y_test = self.test['Z4-3S (W)']
        elif generator == "5S":
            df = self.df5
            # Splitting the training and testing data
            self.train = df.loc[df.index < train_date]
            #self.train = df
            self.test = df.loc[df.index >= test_date]
            X_train = self.train[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
            y_train = self.train['Z5-5S']
            X_test = self.test[['minute', 'hour', 'dayofweek', 'month', 'quarter', 'year','dayofyear']]
            y_test = self.test['Z5-5S']
        else:
            raise ValueError("Unknown training Generator")
        
        # Scale the features using StandardScaler
        scaler = StandardScaler()
        print("....Scaling the initial Generator features.....")
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
        batch_size = self.batch_size  # Adjust batch size as necessary
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        test_dataset = TensorDataset(X_test_scaled, y_test_torch)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # load GPs
        gp_layers_Gm = self.gp_layers_Gm
        gp_layers_Zm = self.gp_layers_Zm
        
        #  Initialize the model with pre-trained GP layers
        model = BayesianNN(gp_layers_Gm, gp_layers_Zm, X_train_scaled[:batch_size], y_train_torch[:batch_size])
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        print("....moving the model and likelihood to device = ", device, ".....")
        # Move model to GPU if available
        model = model.to(device)
        likelihood = likelihood.to(device)

        # Define the training procedure
        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model.output_layer)
        
        # Retrain the model
        trained_model, train_Loss = self.train_model(model, likelihood, optimizer, train_loader)
        if generator == "2S":
            self.gp_layers_Gm[0] = trained_model.gp_layers_Gm[0].state_dict()
            self.Loss_2s.append(train_Loss)
        elif generator == "3S":
            self.gp_layers_Gm[1] = trained_model.gp_layers_Gm[1].state_dict()
            self.Loss_3s.append(train_Loss)
        elif generator == "5S":
            self.gp_layers_Gm[2] = trained_model.gp_layers_Gm[2].state_dict()
            self.Loss_5s.append(train_Loss)
        else:
            raise ValueError("Unknown training Generator")
            
        test_prediction, test_target = self.evaluate_model(model = trained_model, likelihood = likelihood, test_loader = test_loader)
        
        # plot the predicted load
        self.plot_predict(test_x = X_train['hour'], test_y = test_target, preds = test_prediction, generator = generator)
        self.plot_loss(generator = generator)
            
        
        def evaluate_model(self, model, likelihood, test_loader):
            # Make predictions
            model.eval()
            likelihood.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    preds = likelihood(model(batch_x)).mean
                    all_preds.append(preds)
                    all_targets.append(batch_y)

            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            return all_preds, all_targets
        
        # update the model
        
        def plot_predict(self, test_x, test_y, preds, generator):
            # Plotting results
            with torch.no_grad():
                fig, ax = plt.subplots(1, 1, figsize=(10,5))
                ax.plot(test_x.numpy(), test_y.numpy(), 'k*')
                ax.plot(test_x.numpy(), preds.numpy(), 'b')
                ax.legend(['True load', 'Predicted Load'])
                ax.set_xlabel('Hour')
                ax.set_ylabel('Load (W)')
                ax.set_title('Load prediction for July 6th 2015')
                fname = "zonal_prediction_data_generator_"+generator+".png"
                fig.savefig(os.path.join('./../plots/',fname))
                
        def plot_loss(self, generator):
            fig, ax = plt.subplots(1, 1, figsize=(10,5))
            if generator == "2s":
                ax.plot(self.Loss_2s, 'b')
            elif generator == "3s":
                ax.plot(self.Loss_3s, 'b')
            elif generator == "5s":
                ax.plot(self.Loss_5s, 'b')
            else:
                raise ValueError("Unknown training Generator")
            fname = "zonal_loss_generator_"+generator+".png"
            fig.savefig(os.path.join('./../plots/',fname))
                
    
if __name__ == '__main__':
    torch_GPR_zonal(sys.argv[1])
    
    