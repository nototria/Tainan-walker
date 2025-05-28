import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

def load_edges(file_path):
    return pd.read_csv(file_path)

def load_model_and_run(input_csv="paths.csv", batch_size=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load('model.pt', map_location=device)

    # 取出設定
    config = ckpt['config']

    # 1) 重建模型
    model = FCN(
        in_features=config['in_features'],
        hidden_sizes=config['hidden_sizes'],
        n_classes=config['n_classes'],
        dropout=config['dropout']
    ).to(device)

    # 4) 載入狀態
    model.load_state_dict(ckpt['model_state_dict'])

    X_test, y_test = calculate_TOPSIS(input_csv, n_labels=5)

    test_loader = DataLoader(PathDataset(X_test, y_test),
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

    test_preds = torch.cat(test_preds)       # tensor of shape (N,)
    test_labels = torch.cat(test_labels)     # tensor of shape (N,)
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = (test_preds == test_labels).float().mean().item()
    with open('run_val_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"{test_loss:.4f}",f"{test_acc:.4f}"])
    
    # 假設 test_preds 是 shape=(N,) 的 LongTensor
    # 1. 找到最大的 label 值
    max_label = test_preds.max().item()

    # 2. 找出所有預測是這個最大值的樣本 index
    #    nonzero 返回形如 (idx0, idx1, ...)，as_tuple=True 讓它直接回傳一個 tuple
    indices = (test_preds == max_label).nonzero(as_tuple=True)[0]

    # 3. 如果有多筆，隨機選一筆；若只想選第一筆，可直接取 indices[0]
    chosen_idx = indices[torch.randint(len(indices), (1,))].item()

    # 4. 拿到這筆的 label，確認就是 max_label
    chosen_label = test_preds[chosen_idx].item()
    print(f"Chosen index in test set: {chosen_idx}, label: {chosen_label}")

    df = load_edges(input_csv)

    chosen_raw = df[df['path_id']-1 == chosen_idx]
    chosen_raw.to_csv('model_decided_path.csv', index=False)


    



    
def summarize(path, noise_std=0.0):
    """
    對單一路徑 (path) 進行聚合統計，並依 noise_std 加入 Gaussian 噪聲。
    
    noise_std: float
        噪聲的標準差比例，相當於在每個原始值上加上 N(0, noise_std * 原始值)。
        若不想加噪聲，設為 0.0 即可。
    """
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

    # 原始結果
    stats = pd.Series({
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

    if noise_std > 0:
        # 針對每個欄位加上 Gaussian noise
        # scale = noise_std * abs(原始值) + epsilon 保證 scale > 0
        eps = 1e-6
        scales = noise_std * stats.abs() + eps
        noise = np.random.randn(len(stats)) * scales
        stats = stats + noise

    return stats

def calculate_TOPSIS(input_csv="paths_o.csv", n_labels=5):
    # 1. 讀取原始細節資料
    df = load_edges(input_csv)

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

    paths_df['quality_label'] = pd.qcut(
        paths_df['topsis_score'], 
        q=n_labels,                    
        labels=False             
    )
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

def train(paths_df, features, n_epochs=100, batch_size=32, random_seed=42):
    features_array = paths_df[features].values
    target_labels = paths_df['quality_label'].astype(int).values
    X_temp, X_test, y_temp, y_test = train_test_split(
        features_array, target_labels,
        test_size=0.15,
        random_state=random_seed,
        stratify=target_labels
    )

    # 2. 再把剩下的 X_temp, y_temp 切成 Train 與 Val（Train 佔 80%，Val 佔 20%）
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


    # model
    model = FCN(in_features=features_array.shape[1],
                          hidden_sizes=[1024, 512, 256, 128],
                          n_classes=len(np.unique(target_labels)), 
                          dropout=0.3).to(device)
    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',           # 監控的指標：越小越好 → val_loss
        factor=0.5,           # 當觸發降低 LR 時，lr = lr * factor
        patience=3,           # 經過多少個 epoch val_loss 沒下降才降低 lr
        min_lr=1e-6           # lr 的下限
    )

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
        'model_state_dict': model.state_dict(),    # 模型參數
        'config': config                           # 你的設定參數
    }

    torch.save(model, 'model.pt')
    print("model saved to model.pt")

def main():
    df, feat = calculate_TOPSIS(input_csv="traning_paths.csv", n_labels=5)
    feat_arr, tar_labels, trained_model = train(paths_df=df, features=feat, n_epochs=100, batch_size=32, random_seed=42)
    save_model(model=trained_model, X_l=feat_arr, y_l=tar_labels)

if __name__ == '__main__':
    main()