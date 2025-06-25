import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import math

warnings.filterwarnings('ignore')


# 1. Simplified WQI Calculation (unchanged)
def calculate_wqi(row):
    """Simpler more robust WQI calculation"""
    params = {
        'pH': max(0, 1 - abs(row.get('pH Level', 7) - 7) / 3),
        'DO': min(row.get('Dissolved Oxygen (mg/L)', 8) / 15, 1),
        'Ammonia': max(0, 1 - row.get('Ammonia (mg/L)', 0.5) / 2),
        'Nitrate': max(0, 1 - row.get('Nitrate (mg/L)', 0.5) / 10),
        'Phosphate': max(0, 1 - row.get('Phosphate (mg/L)', 0.1) / 0.5),
        'Temp': max(0, 1 - abs(row.get('Surface Water Temp (°C)', 25) - 25) / 15)
    }
    return sum(params.values()) / len(params) * 100

def get_pollutant_level(wqi):
    if wqi >= 91:
        return "Excellent"
    elif wqi >= 71:
        return "Good"
    elif wqi >= 51:
        return "Average"
    elif wqi >= 26:
        return "Fair"
    else:
        return "Poor"


# 2. Hybrid CNN-LSTM Model
class HybridModel(nn.Module):
    def __init__(self, input_size, seq_length=12):
        super().__init__()
        self.seq_length = seq_length

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.4),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        # LSTM Temporal Processor
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

        # WQI Head
        self.fc_wqi = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

        # Pollutants Head (Ammonia, Nitrate, Phosphate)
        self.fc_pollutants = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3)
        )

    def forward(self, x, return_attention=False):
        batch_size = x.size(0)

        cnn_in = x.permute(0, 2, 1)
        cnn_out = self.cnn(cnn_in)
        lstm_in = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_in)

        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        # Use separate heads
        wqi_out = self.fc_wqi(context)           # (batch_size, 1)
        pollutant_out = self.fc_pollutants(context)  # (batch_size, 3)

        # Combine into single output
        output = torch.cat([wqi_out, pollutant_out], dim=1)  # (batch_size, 4)

        if return_attention:
            return output, attn_weights
        return output


# 3. Data Preparation (enhanced)
def prepare_data(data_path, use_climate=False, use_volcanic=False, use_lagged_volcanic=False):
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-15')
    df = df.sort_values('Date').set_index('Date')

    df = df.replace(-999.0, np.nan)
    df = df.clip(lower=0)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            seasonal = df[col].rolling(12, min_periods=1).mean()
            df[col] = df[col].fillna(seasonal)

    df = df.ffill().bfill()

    df['WQI'] = df.apply(calculate_wqi, axis=1)

    # Common engineered features
    df['Rainfall_log'] = np.log1p(df['Rainfall'])
    df['WindDir_sin'] = np.sin(np.radians(df['WindDirection']))
    df['WindDir_cos'] = np.cos(np.radians(df['WindDirection']))
    df['Temp_Δ'] = df['Surface Water Temp (°C)'].rolling(3, min_periods=1).mean()
    df['WindSpeed_Δ'] = df['WindSpeed'].rolling(3, min_periods=1).mean()
    df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    df['Season'] = (df.index.month % 12 + 3) // 3

    # Lag features for water
    for lag in [1, 3, 6, 12]:
        df[f'WQI_lag_{lag}'] = df['WQI'].shift(lag)
        df[f'Temp_lag_{lag}'] = df['Surface Water Temp (°C)'].shift(lag)

    # Rolling
    df['WQI_rolling_3'] = df['WQI'].rolling(3).mean()
    df['WQI_rolling_6'] = df['WQI'].rolling(6).mean()

    # Interaction
    df['DO_pH_ratio'] = df['Dissolved Oxygen (mg/L)'] / (df['pH Level'] + 1e-5)
    df['Ammonia_Nitrate_ratio'] = df['Ammonia (mg/L)'] / (df['Nitrate (mg/L)'] + 1e-5)
    df['Temp_RH_product'] = df['Surface Water Temp (°C)'] * df['RH']
    df['WQI_temp_corr'] = df['WQI_lag_1'] * df['Temp_lag_1']
    df['Rainfall_Wind'] = df['Rainfall_log'] * df['WindSpeed']

    # Base water features
    water_features = [
        'Surface Water Temp (°C)', 'pH Level',
        'Dissolved Oxygen (mg/L)', 'Ammonia (mg/L)',
        'Nitrate (mg/L)', 'Phosphate (mg/L)',
        'DO_pH_ratio', 'Ammonia_Nitrate_ratio',
        'Temp_RH_product', 'WQI_temp_corr',
        'WQI_lag_1', 'WQI_lag_3', 'WQI_lag_12',
        'Temp_lag_1', 'Temp_lag_12',
        'WQI_rolling_3', 'WQI_rolling_6',
    ]

    # Climate features
    climate_features = [
        'Rainfall_log', 'Tmax', 'Tmin', 'RH',
        'WindSpeed', 'WindDir_sin', 'WindDir_cos',
        'Temp_Δ', 'WindSpeed_Δ', 'Month_sin', 'Month_cos',
        'Rainfall_Wind'
    ]

    # Volcanic features
    volcanic_features = []
    if use_volcanic:
        if 'SO2 Flux (t/d)' in df.columns:
            df['SO2'] = df['SO2 Flux (t/d)']
            df['SO2_log'] = np.log1p(df['SO2'])
            volcanic_features.append('SO2_log')
        if 'CO2 Flux (t/d)' in df.columns:
            df['CO2'] = df['CO2 Flux (t/d)']
            df['CO2_log'] = np.log1p(df['CO2'])
            volcanic_features.append('CO2_log')

    # Final feature selection
    if use_climate and not use_volcanic:
        # Match exact feature set from your earlier "water + climate only" run
        features = [
            'DO_pH_ratio', 'Ammonia_Nitrate_ratio', 'Temp_RH_product',
            'WQI_temp_corr', 'Rainfall_Wind',
            'Surface Water Temp (°C)', 'pH Level',
            'Dissolved Oxygen (mg/L)', 'Ammonia (mg/L)',
            'Nitrate (mg/L)', 'Phosphate (mg/L)',
            'Rainfall_log', 'Tmax', 'Tmin', 'RH',
            'WindSpeed', 'WindDir_sin', 'WindDir_cos',
            'Temp_Δ', 'WindSpeed_Δ', 'Month_sin', 'Month_cos',
            'WQI_lag_1', 'WQI_lag_3', 'WQI_lag_12',
            'Temp_lag_1', 'Temp_lag_12',
            'WQI_rolling_3', 'WQI_rolling_6',
        ]
    else:
        # Dynamically build based on user input (for volcanic or water-only)
        features = water_features[:]
        if use_climate:
            features += climate_features
        if use_volcanic:
            features += volcanic_features

    df = df.dropna(subset=features + ['WQI'])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    lookback = 24
    X, y, dates = [], [], []
    for i in range(lookback, len(df)):
        X.append(scaled[i - lookback:i])
        y.append([
            df['WQI'].iloc[i],
            df['Ammonia (mg/L)'].iloc[i],
            df['Nitrate (mg/L)'].iloc[i],
            df['Phosphate (mg/L)'].iloc[i]
        ])
        dates.append(df.index[i])

    y = torch.FloatTensor(y)

    split = int(0.8 * len(X))
    X_train = torch.FloatTensor(X[:split])
    y_train = y[:split]
    X_test = torch.FloatTensor(X[split:])
    y_test = y[split:]

    def add_noise(tensor, noise_level=0.02):
        return tensor + noise_level * torch.randn_like(tensor)

    X_train = add_noise(X_train)

    return X_train, y_train, X_test, y_test, scaler, features, df, dates[split:]


# 4. Enhanced Training Function
def train_model(model, X_train, y_train, X_test, y_test, param_choice, epochs=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0002, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float('inf')
    best_r2 = -float('inf')
    train_losses, val_losses = [], []
    early_stop_counter = 0
    early_stop_patience = 50

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        model.train()
        batch_loss = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)  # shape: [B, 4]
            loss = criterion(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_loss.append(loss.item())

        train_loss = np.mean(batch_loss)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test.to(device))  # shape: [N, 4]
            val_loss = criterion(val_preds, y_test.to(device))
            val_losses.append(val_loss.item())

            y_true = y_test.cpu().numpy()
            y_pred = val_preds.cpu().numpy()

            # Calculate R² for all 4 targets
            val_r2s = [r2_score(y_true[:, i], y_pred[:, i]) for i in range(4)]

        scheduler.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                  f"R²: WQI={val_r2s[0]:.4f}, NH₃={val_r2s[1]:.4f}, NO₃={val_r2s[2]:.4f}, PO₄={val_r2s[3]:.4f}")

        if val_loss < best_loss or max(val_r2s) > best_r2:
            best_loss = val_loss
            best_r2 = max(val_r2s)
            torch.save(model.state_dict(), f'model_{param_choice}.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(f'model_{param_choice}.pth'))

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_preds = model(X_test).cpu().numpy()
        final_true = y_test.cpu().numpy()
        final_r2s = [r2_score(final_true[:, i], final_preds[:, i]) for i in range(4)]
        final_mae = [mean_absolute_error(final_true[:, i], final_preds[:, i]) for i in range(4)]
        final_rmse = [math.sqrt(mean_squared_error(final_true[:, i], final_preds[:, i])) for i in range(4)]

    print("\nFinal Performanceeee:")
    targets = ['WQI', 'Ammonia', 'Nitrate', 'Phosphate']
    for i in range(4):
        print(f"{targets[i]} - MAE: {final_mae[i]:.2f}, RMSE: {final_rmse[i]:.2f}, R²: {final_r2s[i]:.4f}")

    # Optional: Plot WQI only
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return model.to('cpu'), train_losses, val_losses


# 5. Prediction Function (enhanced)
def predict(model, df, scaler, features, date):
    try:
        date = pd.to_datetime(date).replace(day=15)
        lookback = model.seq_length

        if date < df.index[0] + pd.DateOffset(months=lookback):
            raise ValueError(f"Need data from at least {lookback} months before {date.date()}")

        # Estimate monthly trend and seasonal components
        monthly_trend = df[features].diff(12).mean() / 12
        seasonal_means = df.groupby(df.index.month)[features].mean()
        global_mean = df[features].mean()
        seasonal_adjustment = seasonal_means - global_mean

        # Future projection handling
        if date > df.index[-1]:
            print(f"Warning: Projecting into future beyond {df.index[-1].date()}")
            months_ahead = (date.year - df.index[-1].year) * 12 + (date.month - df.index[-1].month)
            base_point = df.iloc[-1][features].copy()

            seq = []
            for i in range(lookback):
                months_offset = months_ahead - (lookback - i)
                future_month = (df.index[-1] + pd.DateOffset(months=months_offset)).month
                point = base_point + monthly_trend * months_offset
                point += seasonal_adjustment.loc[future_month]
                noise = df[features].std() * 0.1 * np.random.randn(len(features))
                point += noise
                seq.append(point)
        else:
            last_idx = df.index.get_loc(date)
            seq = [df.iloc[last_idx - lookback + i][features].copy() for i in range(lookback)]

        seq_scaled = scaler.transform(np.array(seq))
        print(f"Predicting for: {date.strftime('%Y-%m')}")

        # Get all 4 predictions
        with torch.no_grad():
            output = model(torch.FloatTensor(seq_scaled).unsqueeze(0))
            output = output.cpu().numpy().flatten()

        wqi, ammonia, nitrate, phosphate = output
        wqi_clipped = max(0, min(100, wqi))
        level = get_pollutant_level(wqi_clipped)

        print(f"\n🧪 Predicted Values:")
        print(f"  WQI      : {wqi_clipped:.2f} → {level}")
        print(f"  Ammonia  : {ammonia:.4f} mg/L")
        print(f"  Nitrate  : {nitrate:.4f} mg/L")
        print(f"  Phosphate: {phosphate:.4f} mg/L")

        # Show error if actual data exists
        if date in df.index:
            actual = df.loc[date]
            print(f"\n📊 Actual Values:")
            print(f"  WQI      : {actual['WQI']:.2f} | Error: {abs(wqi_clipped - actual['WQI']):.2f}")
            print(f"  Ammonia  : {actual['Ammonia (mg/L)']:.4f} | Error: {abs(ammonia - actual['Ammonia (mg/L)']):.4f}")
            print(f"  Nitrate  : {actual['Nitrate (mg/L)']:.4f} | Error: {abs(nitrate - actual['Nitrate (mg/L)']):.4f}")
            print(f"  Phosphate: {actual['Phosphate (mg/L)']:.4f} | Error: {abs(phosphate - actual['Phosphate (mg/L)']):.4f}")

        return {
            "WQI": wqi_clipped,
            "Ammonia": ammonia,
            "Nitrate": nitrate,
            "Phosphate": phosphate
        }

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None


# 6. Main Execution
def main():
    def set_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(42)

    # ─── User Input ──────────────────────────────
    print("Choose feature configuration:")
    print("a. Water Parameters only")
    print("b. Water + Climate Parameters")
    print("c. Water + Volcanic Parameters")
    print("d. All Parameters")
    param_choice = input("Select (a/b/c/d): ").strip().lower()

    use_climate = False
    use_volcanic = False

    if param_choice == 'b':
        use_climate = True
    elif param_choice == 'c':
        use_volcanic = True
    elif param_choice == 'd':
        use_climate = True
        use_volcanic = True

    # ─── Data Prep ──────────────────────────────
    print("\nPreparing data...")
    X_train, y_train, X_test, y_test, scaler, features, df, test_dates = prepare_data(
        'water_quality_data.csv',
        use_climate=use_climate,
        use_volcanic=use_volcanic,
    )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Feature count: {len(features)}")

    print("\nInitializing model...")
    model = HybridModel(input_size=len(features), seq_length=X_train.shape[1])
    print(model)

    print("\nTraining model...")
    model, train_losses, val_losses = train_model(model, X_train, y_train, X_test, y_test, param_choice, epochs=500)

    with torch.no_grad():
        test_preds = model(X_test).numpy()  # shape: (N, 4)
        test_true = y_test.numpy()

    # Plot all 4 targets: WQI, Ammonia, Nitrate, Phosphate
    targets = ['WQI', 'Ammonia', 'Nitrate', 'Phosphate']
    y_test_np = y_test.numpy()
    test_preds_np = model(X_test).detach().numpy()

    plt.figure(figsize=(14, 10))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(test_dates, y_test_np[:, i], label=f'Actual {targets[i]}')
        plt.plot(test_dates, test_preds_np[:, i], label=f'Predicted {targets[i]}', linestyle='--')
        plt.title(f'{targets[i]}: Actual vs Predicted')
        plt.xlabel('Date')
        plt.ylabel(targets[i])
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('all_predictions.png')
    plt.show()

    # ─── Interactive Prediction ────────────────
    while True:
        date = input("\nEnter prediction date (YYYY-MM-DD) or 'q' to quit: ").strip()
        if date.lower() == 'q':
            break
        pred = predict(model, df, scaler, features, date)
        if pred is not None:
            wqi = pred["WQI"]
            level = get_pollutant_level(wqi)
            print(f"Predicted WQI: {wqi:.2f} → {level}")
            date_obj = pd.to_datetime(date).replace(day=15)
            if date_obj in df.index:
                actual = df.loc[date_obj, 'WQI']
                print(f"Actual WQI: {actual:.2f} | Error: {abs(wqi - actual):.2f}")


if __name__ == "__main__":
    main()