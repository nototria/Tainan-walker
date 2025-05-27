import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import train_test_split
import csv
from sklearn.metrics import accuracy_score

# 1. 讀取原始細節資料
df = pd.read_csv("paths.csv")

# 2. 聚合每條路 (path_id) 的 segment-level 特徵
def summarize(path):
    seg = path[['distance', 'green_ratio', 'shade', 'pavement_ratio']]
    total_dist = seg['distance'].sum()
    wavg = lambda x: (x * seg['distance']).sum() / total_dist
    wvar = lambda x: (seg['distance'] * (x - wavg(x))**2).sum() / total_dist

    start_x = path.iloc[0]['start_x']
    start_y = path.iloc[0]['start_y']
    end_x = path.iloc[-1]['end_x']
    end_y = path.iloc[-1]['end_y']

    straight_dist = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    avg_segment_length = total_dist / len(seg)

    return pd.Series({
        'start_x': start_x,
        'start_y': start_y,
        'end_x':   end_x,
        'end_y':   end_y,
        'total_distance': total_dist,
        'straightness': straight_dist / total_dist if total_dist > 0 else 0,
        'avg_segment_length': avg_segment_length,
        
        # Weighted means
        'green_wmean': wavg(seg['green_ratio']),
        'shade_wmean': wavg(seg['shade']),
        'pavement_wmean': wavg(seg['pavement_ratio']),
        
        # Weighted variances
        'green_wvar': wvar(seg['green_ratio']),
        'shade_wvar': wvar(seg['shade']),
        'pavement_wvar': wvar(seg['pavement_ratio']),
        
        # Simple statistics
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

        # Segment count
        'segment_count': len(path)
    })

paths_df = (
    df.groupby("path_id")
      .apply(summarize, include_groups=False)
      .reset_index()
)

features = [
    'green_wmean', 'shade_wmean', 'pavement_wmean',        # 越大越好
    'green_wvar', 'shade_wvar', 'pavement_wvar',           # 越小越好
    'green_std', 'shade_std', 'pavement_std',              # 越小越好（穩定性）
    'green_skew', 'shade_skew', 'pavement_skew',           # 視情況而定，這裡預設絕對值越小越好（偏態越低越穩定）
    'green_median', 'shade_median', 'pavement_median',     # 越大越好
    'green_max', 'shade_max', 'pave_max',                  # 越大越好
    'green_min', 'shade_min', 'pave_min',                  # 越大越好（代表底線不差）
    'straightness',                                        # 越接近直線越好（視需求）
    'avg_segment_length',                                  # 可視為中性或越小越好（穩定度高）
    'segment_count'                                        # 視情況，這裡假設越多越穩定
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

    'green_skew': False,  # 越不偏越好
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

    'straightness': True,            # 越直越好
    'avg_segment_length': False,     # 越穩定越好 → 越小越好
    'segment_count': True            # 越多越好（資訊量大）
}

# 3.1 正規化
X = paths_df[features].values
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

# 3.2 權重 (等權)
weights = np.ones(len(features)) / len(features)
X_weighted = X_norm * weights

# 3.3 理想解與反理想解
ideal_pos = np.array([
    X_weighted[:, i].max() if benefit[f] else X_weighted[:, i].min()
    for i, f in enumerate(features)
])
ideal_neg = np.array([
    X_weighted[:, i].min() if benefit[f] else X_weighted[:, i].max()
    for i, f in enumerate(features)
])

# 3.4 計算距離
d_pos = np.linalg.norm(X_weighted - ideal_pos, axis=1)
d_neg = np.linalg.norm(X_weighted - ideal_neg, axis=1)

# 3.5 TOPSIS 分數
paths_df['topsis_score'] = d_neg / (d_pos + d_neg)

# 4. 根據 TOPSIS 分數離散化為三級品質標籤 (0=差,1=中,2=好)
q1 = paths_df['topsis_score'].quantile(1/3)
q2 = paths_df['topsis_score'].quantile(2/3)
def assign_label(s):
    if s <= q1: return 0
    elif s <= q2: return 1
    else: return 2

paths_df['quality_label'] = paths_df['topsis_score'].apply(assign_label)


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

class MLPClassifier(nn.Module):
    def __init__(self, in_features, hidden_sizes=[256, 128, 64, 32], n_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            prev = h
        layers += [nn.Linear(prev, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, in_features)
        return self.net(x)  # 未經 softmax 的 logits

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)                   # (batch, n_classes)
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

X_l = paths_df[features].values
y_l = paths_df['quality_label'].astype(int).values
X_train, X_val, y_train, y_val = train_test_split(
    X_l, y_l,
    train_size=0.8,
    random_state=42,
    stratify=y_l
)

batch_size = 32
ds_l = PathDataset(X_l, y_l)
loader_l = DataLoader(PathDataset(X_l, y_l), batch_size=batch_size, shuffle=True)

loader_train = DataLoader(PathDataset(X_train, y_train),
                          batch_size=batch_size, shuffle=True)
loader_val   = DataLoader(PathDataset(X_val,   y_val),
                          batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MLPClassifier(in_features=X_l.shape[1], 
                      hidden_sizes=[128, 64], 
                      n_classes=len(np.unique(y_l)), 
                      dropout=0.5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

n_epochs = 100

with open('train_val_log.csv', 'w', newline='') as f:
    writer = csv.writer(f); writer.writerow(['epoch','train_loss','train_acc','val_loss','val_acc'])

for epoch in range(1, n_epochs+1):
    loss = train_epoch(model, loader_train, criterion, optimizer, device)
        # 計算 Training Accuracy
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in loader_l:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = torch.argmax(model(Xb), dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    train_acc = accuracy_score(np.concatenate(all_labels), np.concatenate(all_preds))

    val_loss, val_acc = validate_epoch(model, loader_val, criterion, device)

    print(f"Epoch {epoch}/{n_epochs}  "
          f"Train: loss={loss:.4f}, acc={train_acc:.4f}  "
          f"Val:   loss={val_loss:.4f}, acc={val_acc:.4f}")

    with open('train_val_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, loss, train_acc, val_loss, val_acc])
