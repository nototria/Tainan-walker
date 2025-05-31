import csv
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def load_edges(file_path):
    return pd.read_csv(file_path)

def load_model_and_run(output_csv, input_csv="paths.csv", batch_size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load('../model.pt', map_location=device)

    config = ckpt['config']

    model = FCN(
        in_features=config['in_features'],
        hidden_sizes=config['hidden_sizes'],
        n_classes=config['n_classes'],
        dropout=config['dropout']
    ).to(device)

    model.load_state_dict(ckpt['model_state_dict'])

    paths_df, features = statistics(input_csv, n_labels=5)

    features_array = paths_df[features].values
    target_labels = paths_df['quality_label'].astype(int).values

    test_loader = DataLoader(PathDataset(features_array, target_labels),
                         batch_size=batch_size,
                         shuffle=False)

    criterion = nn.CrossEntropyLoss()

    model.eval()
    running_loss = 0.0
    test_preds, test_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            running_loss += loss.item() * Xb.size(0)

            preds = torch.argmax(logits, dim=1)
            test_preds.append(preds.cpu())
            test_labels.append(yb.cpu())

    test_preds = torch.cat(test_preds)       
    test_labels = torch.cat(test_labels)     


    max_label = test_preds.max().item()

    indices = (test_preds == max_label).nonzero(as_tuple=True)[0]

    chosen_idx = indices[torch.randint(len(indices), (1,))].item()

    chosen_label = test_preds[chosen_idx].item()
    print(f"Chosen index in test set: {chosen_idx}, label: {chosen_label}")

    df = load_edges(input_csv)

    chosen_raw = df[df['path_id']-1 == chosen_idx]
    chosen_raw.to_csv(output_csv, index=False)

def summarize(path, noise_std=0.0):
    seg = path[['distance', 'green_ratio', 'shade', 'pavement_ratio']]
    total_dist = seg['distance'].sum()
    wavg = lambda x: (x * seg['distance']).sum() / total_dist
    wvar = lambda x: (seg['distance'] * (x - wavg(x))**2).sum() / total_dist

    start_x = path.iloc[0]['start_x']
    start_y = path.iloc[0]['start_y']
    end_x   = path.iloc[-1]['end_x']
    end_y   = path.iloc[-1]['end_y']

    straight_dist = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    avg_segment_length = total_dist / len(seg)

    stats = pd.Series({
        'start_x': start_x,
        'start_y': start_y,
        'end_x':   end_x,
        'end_y':   end_y,
        'total_distance': total_dist,
        'straightness': straight_dist / total_dist if total_dist > 0 else 0,
        'avg_segment_length': avg_segment_length,

        'green_wmean': wavg(seg['green_ratio']),
        'shade_wmean': wavg(seg['shade']),
        'pavement_wmean': wavg(seg['pavement_ratio']),

        'green_wvar': wvar(seg['green_ratio']),
        'shade_wvar': wvar(seg['shade']),
        'pavement_wvar': wvar(seg['pavement_ratio']),

        'green_max': seg['green_ratio'].max(),
        'green_min': seg['green_ratio'].min(),
        'green_median': seg['green_ratio'].median(),
        'green_std': seg['green_ratio'].std(),
        'green_skew': seg['green_ratio'].skew(),

        'shade_max': seg['shade'].max(),
        'shade_min': seg['shade'].min(),
        'shade_median': seg['shade'].median(),
        'shade_std': seg['shade'].std(),
        'shade_skew': seg['shade'].skew(),

        'pave_max': seg['pavement_ratio'].max(),
        'pave_min': seg['pavement_ratio'].min(),
        'pavement_median': seg['pavement_ratio'].median(),
        'pavement_std': seg['pavement_ratio'].std(),
        'pavement_skew': seg['pavement_ratio'].skew(),

        'segment_count': len(path)
    })

    if noise_std > 0:

        eps = 1e-6
        scales = noise_std * stats.abs() + eps
        noise = np.random.randn(len(stats)) * scales
        stats = stats + noise

    return stats

def statistics(input_csv="paths_o.csv", n_labels=5):

    df = load_edges(input_csv)
    paths_df = (
        df.groupby("path_id")
          .apply(summarize, include_groups=False)
          .reset_index()
    )

    features = [
        'green_wmean', 'shade_wmean', 'pavement_wmean',        
        'green_wvar', 'shade_wvar', 'pavement_wvar',           
        'green_std', 'shade_std', 'pavement_std',              
        'green_skew', 'shade_skew', 'pavement_skew',           
        'green_median', 'shade_median', 'pavement_median',     
        'green_max', 'shade_max', 'pave_max',                  
        'green_min', 'shade_min', 'pave_min',                  
        'straightness',                                        
        'avg_segment_length',                                  
        'segment_count'                                        
    ]

    benefit = {
        'green_wmean': True,
        'shade_wmean': True,
        'pavement_wmean': True,

        'green_wvar': False,
        'shade_wvar': False,
        'pavement_wvar': False,

        'green_std': False,
        'shade_std': False,
        'pavement_std': False,

        'green_skew': False,  
        'shade_skew': False,
        'pavement_skew': False,

        'green_median': True,
        'shade_median': True,
        'pavement_median': True,

        'green_max': True,
        'shade_max': True,
        'pave_max': True,

        'green_min': True,
        'shade_min': True,
        'pave_min': True,

        'straightness': True,            
        'avg_segment_length': False,     
        'segment_count': True            
    }

    missing = [f for f in features if f not in paths_df.columns]
    if len(missing) == len(features):
        print("No paths found. close.")
        sys.exit(0)   

    X = paths_df[features].values
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)

    weights = np.ones(len(features)) / len(features)
    X_weighted = X_norm * weights

    ideal_pos = np.array([
        X_weighted[:, i].max() if benefit[f] else X_weighted[:, i].min()
        for i, f in enumerate(features)
    ])
    ideal_neg = np.array([
        X_weighted[:, i].min() if benefit[f] else X_weighted[:, i].max()
        for i, f in enumerate(features)
    ])

    d_pos = np.linalg.norm(X_weighted - ideal_pos, axis=1)
    d_neg = np.linalg.norm(X_weighted - ideal_neg, axis=1)

    paths_df['topsis_score'] = d_neg / (d_pos + d_neg)

    valid = paths_df['topsis_score'].notna()
    paths_df.loc[valid, 'quality_label'] = pd.qcut(
        paths_df.loc[valid, 'topsis_score'],
        labels=False,
        q=n_labels,                            
        duplicates='drop'               
    )

    paths_df['quality_label'] = paths_df['quality_label'].astype(object)
    paths_df.loc[~valid, 'quality_label'] = 0

    print(paths_df)
    return paths_df, features

class PathDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

class FCN(nn.Module):
    def __init__(self, in_features, hidden_sizes=
                 [1024, 512, 256, 128], n_classes=5, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)                   
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * Xb.size(0)
    return running_loss / len(loader.dataset)

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss = criterion(logits, yb)
            running_loss += loss.item() * Xb.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))
    return avg_loss, acc

def train(paths_df, features, n_epochs=100, batch_size=32, random_seed=42):
    features_array = paths_df[features].values
    target_labels = paths_df['quality_label'].astype(int).values
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_array, target_labels,
        test_size=0.15,
        random_state=random_seed,
        stratify=target_labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        train_size=0.8,
        random_state=random_seed,
        stratify=y_temp
    )

    loader_train = DataLoader(PathDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    loader_val   = DataLoader(PathDataset(X_val,   y_val),
                              batch_size=batch_size, shuffle=False)

    loader_test  = DataLoader(PathDataset(X_test,  y_test),
                              batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FCN(in_features=features_array.shape[1],
                          hidden_sizes=[1024, 512, 256, 128],
                          n_classes=len(np.unique(target_labels)), 
                          dropout=0.3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           
        factor=0.5,           
        patience=3,           
        min_lr=1e-6           
    )

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    with open('train_val_log.csv', 'w', newline='') as f:
        writer = csv.writer(f); writer.writerow(['epoch','train_loss','train_acc','val_loss','val_acc'])

    for epoch in range(1, n_epochs+1):
        loss = train_epoch(model, loader_train, criterion, optimizer, device)

        val_loss, val_acc = validate_epoch(model, loader_val, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}/{n_epochs}  "
              f"Train: loss={loss:.4f},   "
              f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

        with open('train_val_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss, f"{val_loss:.4f}", f"{val_acc:.4f}"])

    test_loss, test_acc = validate_epoch(model, loader_test, criterion, device)
    print(f"*** Test Result ***  loss={test_loss:.4f}, acc={test_acc:.4f}")

    return features_array, target_labels, model

def save_model(model, X_l, y_l):

    config = {
        'in_features': X_l.shape[1],
        'hidden_sizes': [1024, 512, 256, 128],
        'n_classes': len(np.unique(y_l)),
        'dropout': 0.3,
        'random_seed': 42
    }

    model = {
        'model_state_dict': model.state_dict(),    
        'config': config                           
    }

    torch.save(model, 'model.pt')
    print("model saved to model.pt")

def main():
    df, feat = statistics(input_csv="training_paths.csv", n_labels=5)
    feat_arr, tar_labels, trained_model = train(paths_df=df, features=feat, n_epochs=100, batch_size=32, random_seed=42)
    save_model(model=trained_model, X_l=feat_arr, y_l=tar_labels)

if __name__ == '__main__':
    main()