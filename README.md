# 🚀 Credit Card Fraud Detection using LSTM

This project implements **credit card fraud detection** using a **Long Short-Term Memory (LSTM)** neural network. It is trained on the **Kaggle Credit Card Fraud Detection dataset** to classify transactions as **legitimate (0) or fraudulent (1)**.

---

## 📊 Dataset Information

📌 **Dataset Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
📌 **Total Transactions:** 284,807  
📌 **Fraudulent Cases:** 0.17% (Highly imbalanced dataset)  
📌 **Features:**
- `Time`: Time in seconds since the first transaction  
- `V1 - V28`: Anonymized PCA-transformed features  
- `Amount`: Transaction amount  
- `Class`: **Target Variable** (0 = Legitimate, 1 = Fraudulent)  

---

## ⚙️ Installation & Setup

### 🔹 Step 1: Clone the Repository
```sh
git clone https://github.com/toshankanwar/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

```
### 🔹 Step 2: Install Dependencies
```sh
pip install -r requirements.txt
```
### 🔹 Step 3: Train the LSTM Model
```sh
python train_lstm.py
```
### 🔹 Step 4: Test the Model
```sh
python test_lstm.py
```

## 📂 Project Structure
📁 Credit-Card-Fraud-Detection
│── 📄 train_lstm.py        # Train & save the LSTM model
│── 📄 test_lstm.py         # Load & test the trained model
│── 📄 lstm_fraud_model.h5  # Trained LSTM model file
│── 📄 requirements.txt     # Dependencies required for the project
│── 📄 README.md            # Project documentation
│── 📁 dataset
│   ├── 📄 creditcard.csv   # Credit card fraud dataset

## 📂 🛠 Model Architecture
✔️ LSTM (Long Short-Term Memory) neural network

✔️ Dropout layers to prevent overfitting

✔️ Binary classification using sigmoid activation

✔️ Trained with Adam optimizer & Binary Crossentropy loss

## 📊 Performance Metrics

| Metric    | Value  |
|-----------|--------|
| **Accuracy**  | 99.2%  |
| **Precision** | 93.4%  |
| **Recall**    | 88.7%  |
| **F1-Score**  | 91.0%  |

## 📂 🚀 Features & Technologies
Python (NumPy, Pandas, Matplotlib, Seaborn)

Machine Learning (SMOTE for class balancing, Logistic Regression)

Deep Learning (TensorFlow, LSTM Model)

Imbalanced Data Handling (Oversampling techniques)

## 📂 📬 Future Enhancements
✔️ Improve Class Balancing (SMOTE variations, Generative models)

✔️ Deploy as an API (FastAPI, Flask, or Django)

✔️ Integrate Real-Time Monitoring

## 📂 📜 License
This project is open-source under the MIT License.

## 🙌 Contributing
Feel free to fork this repository and contribute with pull requests.

## 📞 Contact

- 📧 Email: [contact@toshankanwar.website](mailto:contact@toshankanwar.website)  
- 🌐 Website: [toshankanwar.website](https://toshankanwar.website/) 
- 🔗 Github: [github.com/toshankanwar](https://github.com/toshankanwar)  




