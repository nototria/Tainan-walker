import csv
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import lightgbm as lgb
import joblib
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


def load_edges(file_path):
    return pd.read_csv(file_path)

def load_model_and_run(output_csv, input_csv="paths.csv", batch_size=10):
    bst = lgb.Booster(model_file="lgbm_model.txt")
    
    paths_df, features = calculate_TOPSIS(input_csv, 5)
    X = paths_df[features].values

    proba = bst.predict(X)
    preds = np.argmax(proba, axis=1)

    max_label = preds.max()
    indices = np.where(preds == max_label)[0]
    
    chosen_idx = np.random.choice(indices)
    chosen_label = preds[chosen_idx]
    print(f"Chosen index in test set: {chosen_idx}, label: {chosen_label}")

    df = load_edges(input_csv)
    chosen_raw = df[df['path_id'] == (chosen_idx + 1)]
    
    chosen_raw.to_csv(output_csv, index=False)
    print(f"Saved chosen path edges to {output_csv}")

def summarize(path, noise_std=0.01):
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

def calculate_TOPSIS(input_csv="paths_o.csv", n_labels=5):

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
    def __init__(self, in_features, hidden_sizes=[1024, 512, 256, 128], n_classes=10, dropout=0.3):
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


    model = LGBMClassifier(
      objective='multiclass',
      num_class=len(np.unique(target_labels)),
      learning_rate=0.05,
      n_estimators=100,
      num_leaves=31,
      colsample_bytree=0.8,
      subsample=0.8,
      reg_alpha=0.1,
      reg_lambda=0.1,
      random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_names=['valid'],
        eval_metric=['multi_logloss', 'multi_error'],
    )

    X_val = pd.DataFrame(X_val, columns=features)
    y_pred = model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

    return features_array, target_labels, model

def save_model(model):
    model.booster_.save_model("lgbm_model.txt")
    

def main():
    df, feat = calculate_TOPSIS(input_csv="traning_paths.csv", n_labels=5)
    feat_arr, tar_labels, trained_model = train(paths_df=df, features=feat, n_epochs=100, batch_size=32, random_seed=42)
    save_model(model=trained_model)

if __name__ == '__main__':
    main()