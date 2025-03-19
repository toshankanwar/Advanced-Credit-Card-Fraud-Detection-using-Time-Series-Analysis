import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained LSTM model
model = load_model("lstm_fraud_model.h5")
print("✅ Model loaded successfully!")

# Define feature names (must match training data)
feature_names = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                 "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
                 "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount_Norm"]

# 5 Sample Transactions
sample_data = [
    {"V1": 1.2, "V2": -0.3, "V3": 2.5, "V4": 1.1, "V5": -3.0, "V6": 0.5, "V7": 1.0, "V8": 0.2, "V9": -1.2, "V10": 0.7, "V11": 1.5, "V12": -2.1, "V13": 0.6, "V14": 0.8, "V15": -0.9, "V16": 1.3, "V17": -0.5, "V18": 2.2, "V19": 1.1, "V20": -1.4, "V21": 0.9, "V22": 0.4, "V23": 1.7, "V24": -0.6, "V25": 0.5, "V26": 0.2, "V27": -0.3, "V28": 1.1, "Amount_Norm": 0.5},
    {"V1": -2.1, "V2": 0.5, "V3": -1.5, "V4": -0.8, "V5": 2.7, "V6": -1.4, "V7": 0.9, "V8": -0.5, "V9": 1.1, "V10": -0.3, "V11": 2.2, "V12": 1.5, "V13": -1.1, "V14": 0.6, "V15": 1.7, "V16": -2.0, "V17": 0.8, "V18": -1.3, "V19": 2.4, "V20": -0.9, "V21": 0.7, "V22": 1.3, "V23": -0.4, "V24": 1.9, "V25": -1.1, "V26": 0.5, "V27": -0.7, "V28": 1.4, "Amount_Norm": 1.2},
    {"V1": 0.5, "V2": -1.2, "V3": 0.9, "V4": -0.4, "V5": 2.3, "V6": -0.7, "V7": 1.2, "V8": 0.4, "V9": -0.8, "V10": 0.6, "V11": 2.0, "V12": -1.4, "V13": 0.8, "V14": -0.2, "V15": 1.5, "V16": -1.7, "V17": 0.9, "V18": -2.2, "V19": 1.3, "V20": -0.5, "V21": 1.1, "V22": -1.0, "V23": 0.7, "V24": -0.3, "V25": 1.8, "V26": -0.9, "V27": 0.5, "V28": -1.6, "Amount_Norm": 0.8},
    {"V1": -1.5, "V2": 0.7, "V3": -2.0, "V4": 0.3, "V5": -1.2, "V6": 1.6, "V7": -0.7, "V8": 2.3, "V9": 1.9, "V10": -0.8, "V11": 0.4, "V12": 1.5, "V13": -0.9, "V14": 2.1, "V15": -1.4, "V16": 0.6, "V17": -2.5, "V18": 1.3, "V19": -0.7, "V20": 0.9, "V21": -1.0, "V22": 2.5, "V23": -0.3, "V24": 1.2, "V25": -0.6, "V26": 0.7, "V27": -1.8, "V28": 0.4, "Amount_Norm": 0.6},
    {"V1": 2.1, "V2": -0.5, "V3": 1.3, "V4": -1.0, "V5": 2.0, "V6": -0.8, "V7": 1.5, "V8": 0.3, "V9": -0.6, "V10": 0.9, "V11": -1.2, "V12": 2.3, "V13": -0.5, "V14": 1.4, "V15": -1.6, "V16": 0.7, "V17": -0.9, "V18": 2.1, "V19": -0.4, "V20": 0.8, "V21": -1.5, "V22": 1.3, "V23": -0.7, "V24": 2.0, "V25": -0.2, "V26": 0.9, "V27": -1.3, "V28": 1.7, "Amount_Norm": 0.9}
]

# Convert input data to DataFrame
sample_df = pd.DataFrame(sample_data, columns=feature_names)

# Reshape data for LSTM (3D input)
sample_lstm = np.reshape(sample_df.values, (sample_df.shape[0], sample_df.shape[1], 1))

# Make predictions
predictions = model.predict(sample_lstm)

# Interpret results
print("\n🔍 Fraud Detection Results:\n")
for i, pred in enumerate(predictions):
    result = "❌ Fraudulent Transaction Detected!" if pred > 0.5 else "✅ Legitimate Transaction."
    print(f"Transaction {i+1}: {result}")
