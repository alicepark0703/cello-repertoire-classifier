from pathlib import Path
from pathlib import Path
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configuration 
FEATURES_CSV = Path()
RANDOM_SEED = 42 # random seed
TEST_SIZE = 0.2 #fraction of source to place in test

#neural network training settings
BATCH_SIZE = 32 #batches 
EPOCHS = 40 #how many times the dataset is seen during training
LEARNING_RATE = 1e-3 #how big a step model takes


#Hidden layer sizes for MLP - smaller = compressing
HIDDEN_1 = 128
HIDDEN_2 = 64

#Dropout to reduce overfitting
DROPOUT = 0.3 #less variability in learning

#use gpu if allowed or else cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#reproduction
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed #torch Generator object
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_sourcegroup(filename : str) ->str:
    """
    extracting og source recording ID from clip filename
    """
    stem = Path(filename).stem
    if "_clip" in stem:
        return stem.split("_clip_")[0] #return everything before
    return stem

#customing dataset for pytorch - wrap in Dataset obj (aka class)
class FeatureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X = feature matrix (num_samples, num_features)
        y = integer class
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype = torch.long)
    
    def __len__(self):
        return len(self.X) #number of dataset samples in order to divide into batches / epoch completion
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] #returning one single sample at defined index (feature, label) as pair
    
    

#defining neural network
# small multilayer perception MLP
#input -> linear -> ReLU -> dropout
#      -> linear -> ReLU -> dropouts
#      -> linear -> class logits
class MLP(nn.Module):
    def __init__(self, input_dim : int, num_classes: int):
        """
        intput_dim = # input features
        num_classes = # output classes 
        """

        super().__init__()

        self.net = nn.Sequential( #chaining layers for continuous data flow

        #first
        nn.Linear(input_dim, HIDDEN_1),
        nn.ReLU(),
        nn.Dropout(DROPOUT),

        #second
        nn.Linear(HIDDEN_1, HIDDEN_2),
        nn.ReLU(),
        nn.Dropout(DROPOUT),

        #output 
        nn.Linear(HIDDEN_2, num_classes)

        )


#training one epoch
def train_epoch(model, loader, criterion, optimizer):
    model.train()

    running_loss = 0 #losses from each batch
    correct = 0 #correct predictions
    total = 0 #total samples seen

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(logits, y_batch)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(logits, dim = 1)
        correct += (preds == y_batch).sum().item() #raw number counting how many are true
        total += y_batch.size(0)

    epoch_loss = running_loss / total
    epoch_accur = correct / total
    return epoch_loss, epoch_accur


#evaluation on test data - same without backpropagation
@torch.no_grad() #no gradient during testing
def evaluation(model, loader, criterion):
    