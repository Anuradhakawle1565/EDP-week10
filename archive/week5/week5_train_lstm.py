import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train():
    from prepare_data import load_close, make_lstm_data

    df = load_close()
    X_train, y_train, X_test, y_test, dates, scaler = make_lstm_data(df, lookback=60)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=25,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    pred_scaled = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(pred_scaled).ravel()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    r2 = r2_score(actual, pred)

    os.makedirs("models", exist_ok=True)
    model.save("models/lstm_model.keras")

    out = pd.DataFrame({"Date": dates, "Actual": actual, "Predicted": pred})
    os.makedirs("data/processed", exist_ok=True)
    out.to_csv("data/processed/lstm_predictions.csv", index=False)

    print(f"LSTM -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {"Model": "LSTM", "MAE": mae, "RMSE": rmse, "R2": r2}, out

if __name__ == "__main__":
    train()
