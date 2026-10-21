"""
make_standin_ae.py - TESTING ONLY.
Trains a tiny autoencoder on non-fraud training rows so you can test src/evaluate.py and src/explain.py
before Member 3 delivers the real autoencoder. Results made with it must NOT be reported or committed.
Run from repo root:  python3 scripts/make_standin_ae.py
"""
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

X = pd.read_csv("processed/X_train.csv")
y = pd.read_csv("processed/y_train.csv").iloc[:, 0]
scaler = joblib.load("models/dnn_improved_scaler.pkl")

Xn = scaler.transform(X[y == 0]).astype("float32")
rng = np.random.default_rng(0)
Xn = Xn[rng.choice(len(Xn), min(200_000, len(Xn)), replace=False)]
d = Xn.shape[1]

ae = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(d,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(d),
])
ae.compile(optimizer="adam", loss="mse")
ae.fit(Xn, Xn, epochs=5, batch_size=512, validation_split=0.1, verbose=1)
ae.save("models/autoencoder_standin.keras")
print("saved models/autoencoder_standin.keras  (STAND-IN, testing only)")
