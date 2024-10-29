import argparse
import warnings
from collections import OrderedDict

from flwr.client import NumPyClient, ClientApp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import numpy as np

import sys
import os

from FL.utils import utils
 
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir) 
 
from Scripts.datagen import Datagen
#from Scripts.lstm import *




# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################
# Get partition id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    choices=[0, 1],
    default=0,
    type=int,
    help="Partition of the dataset divided into 2 iid partitions created artificially.",
)

class main(object):
    def __init__(self, generator=1):
        # Initialize the model
        input_size = 6
        hidden_size = 64
        output_size = 1
        self.net = LSTMModel(input_size, hidden_size, output_size).to(DEVICE)
        self.utils = utils(generator = generator)
        self.trainloader = self.utils.trainloader
        self.testloader = self.utils.testloader


# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, main):
        self.main = main
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.main.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.main.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.main.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.main.utils.train(self.main.net, self.main.trainloader, epochs=30)
        return self.get_parameters(config={}), len(self.main.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = self.main.utils.test(self.main.net, self.main.testloader)
        return loss, len(self.main.testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """Create and return an instance of Flower `Client`."""
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)


# Legacy mode
if __name__ == "__main__":
    from flwr.client import start_client
    main=main(sys.argv[1])
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(main).to_client(),
    )