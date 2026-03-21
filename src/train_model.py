from pathlib import Path
from pathlib import Path
import random
import numpy as np
import pandas as pd

#scikit-learn tools for
# 1) splitting by group
# 2) feature scaling
# 3) converting string label into integer class IDs
# 4) evluation metrics
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
        return len(self.X) #number of dataset samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] #returning one sample at defined index
    
    

#defining neural network
# small multilayer perception MLP
#input -> linear -> ReLU -> dropout
#      -> linear -> ReLU -> dropouts
#      -> linear -> class logits




