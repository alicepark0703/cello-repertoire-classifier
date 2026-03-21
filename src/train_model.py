from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Configuration 
FEATURES_CSV = Path("data/features.csv")
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
    torch.manual_seed(seed) #torch Generator object
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_sourcegroup(filename : str) ->str:
    """
    extracting og source recording ID from clip filename
    """
    stem = Path(filename).stem
    if "_clip_" in stem:
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
    def forward(self, x):
        return self.net(x)


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
    model.eval() #evaluation mode

    running_loss = 0
    correct = 0
    total = 0

    all_pred = []
    all_expect = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        running_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(logits, dim = 1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

        all_pred.extend(preds.cpu().numpy())
        all_expect.extend(y_batch.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_accur = correct / total
    return epoch_loss, epoch_accur, np.array(all_pred), np.array(all_expect)


def main():
    set_seed(RANDOM_SEED) #same seed every time = reproducible

    #load feature csv
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"could not find feature csv")

    df = pd.read_csv(FEATURES_CSV)
    print(f"loaded {len(df)} rows")

    required_col = {"filename", "label"}
    missing = required_col - set(df.columns) #find missing 
    if missing:
        raise ValueError(f"missing required columns {missing}")
    
    #create grouping col
    df["source_group"] = df["filename"].apply(infer_sourcegroup)

    print(f"number of unique source groups = {df['source_group'].nunique}")
    #could print them to debug

    #separate feature columns from metadata
    non_feature_cols = {"filepath", "filename", "label", "source_group"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]

    if len(feature_cols) == 0:
        raise ValueError("no features columns could be found")
    
    X = df[feature_cols].copy() #feature matrix
    y = df["label"].copy() #string labels
    groups = df["source_group"].copy() #source IDs

    #remove bad rows / invalid data
    mask = np.isfinite(X.to_numpy()).all(axis = 1)
    dropped = len(X) - mask.sum()

    if dropped > 0:
        print(f"{dropped} rows have been dropped because invalid feature values exist")

    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    groups = groups.loc[mask].reset_index(drop=True)

    #split by source group
    gss = GroupShuffleSplit(
        n_splits = 1,
        test_size = TEST_SIZE,
        random_state = RANDOM_SEED
    )

    train_idx, test_idx = next(gss.split(X, y, groups = groups))
    X_train_df = X.iloc[train_idx].reset_index(drop = True)
    X_test_df = X.iloc[test_idx].reset_index(drop=True)

    y_train_raw = y.iloc[train_idx].reset_index(drop=True)
    y_test_raw = y.iloc[test_idx].reset_index(drop=True)

    groups_train = groups.iloc[train_idx].reset_index(drop=True)
    groups_test = groups.iloc[test_idx].reset_index(drop=True)

    #test and train data must not overlap
    overlap = set(groups_train) & set(groups_test)
    if overlap:
        raise RuntimeError(f"overlapping groups found in test & train")
    
    print(f"train rows: {len(X_train_df)}")
    print(f"test rows: {len(X_test_df)}")
    print(f"train groups = {groups_train.nunique} & test groups = {groups_test.nunique}")

    #scale features 
    scalar = StandardScaler()

    #mean/std from training data
    X_train = scalar.fit_transform(X_train_df)

    #same transformation to test data
    X_test = scalar.transform(X_test_df)

    #encoding labels
    label_encoder = LabelEncoder()
    
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    num_classes = len(label_encoder.classes_)
    input_dim = X_train.shape[1]


    #wrap data into pytorch datasets / loaders
    train_dataset = FeatureDataset(X_train, np.array(y_train))
    test_dataset = FeatureDataset(X_test, np.array(y_test))

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #model, loss, optimizer
    model = MLP(input_dim=input_dim, num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    #train
    for epoch in range(1, EPOCHS+1):
        train_loss, train_accur = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_accur, _, _ = evaluation(model, test_loader, criterion)
    
        print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"train_loss={train_loss:.4f} train_accur={train_accur:.4f} | "
                f"test_loss={test_loss:.4f} test_accur={test_accur:.4f}"
            )

    #eval
    test_loss, test_accur, y_pred, y_true = evaluation(model, test_loader, criterion)

    print("\n[FINAL TEST RESULTS]")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")

    #per-class precision
    target_names = label_encoder.classes_
    print(classification_report(y_true, y_pred, target_names=target_names))


    #confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("[CONFUSION MATRIX]")
    print(cm)

    asset_dir = Path("assets")
    asset_dir.mkdir(exist_ok = True)

    class_names = list(label_encoder.classes_)

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(asset_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("confusion matrix has been saved to assets/confusion_matrx.png")

    #saving model
    save_obj = {
        "model_state_dict": model.state_dict(),
        "feature_columns": feature_cols,
        "label_classes": list(label_encoder.classes_)
    }

    torch.save(save_obj, "models/models_mlp.pth")
    print("\n[INFO] Saved model to models_mlp.pth")

if __name__ == "__main__":
    main()

