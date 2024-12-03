import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.volume import VolumeWeightedAveragePrice
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Data loading function
def load_data(filepath):
    """Load crypto data from a CSV file."""
    return pd.read_csv(filepath)

# Adding technical indicators
def add_technical_indicators(df):
    """Add MACD, VWAP, and Historical Volatility indicators."""
    # MACD
    macd = MACD(close=df['close'], window_slow=15, window_fast=5, window_sign=7)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    # VWAP
    vwap = VolumeWeightedAveragePrice(
        high=df['high'], low=df['low'], close=df['close'], volume=df['volume']
    )
    df['VWAP'] = vwap.volume_weighted_average_price()

    # Historical Volatility
    window_vol = 10
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df['Volatility'] = df['log_return'].rolling(window=window_vol).std() * np.sqrt(window_vol)

    return df

# Data processing
def preprocess_data(df, features_to_scale):
    """Prepare data by normalizing features and setting the target variable."""
    df = df.dropna()
    df['target'] = (df['log_return'] > 0).astype(int)  # Target: 1 si rendement positif, 0 sinon

    # Normalisation des features
    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df

# Data spliting
def split_data(df, features, target, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# Training
def train_logistic_regression(X_train, y_train):
    """Train a logistic regression model."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

# Evaluation
def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Evaluate the model and print key metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

    return roc_auc

def main(filepath):
    """Main function to run the crypto trading strategy model."""

    df = load_data(filepath)

    df = add_technical_indicators(df)

    features_to_scale = ['Volatility', 'MACD_Diff', 'VWAP']
    df = preprocess_data(df, features_to_scale)

    X_train, X_test, y_train, y_test = split_data(df, features_to_scale, 'target')

    model = train_logistic_regression(X_train, y_train)

    roc_auc = evaluate_model(model, X_test, y_test, threshold=0.6)
    print(f"ROC AUC Score: {roc_auc:.2f}")

# Execution
if __name__ == "__main__":
    filepath = r'C:\Users\Proab\OneDrive\Bureau\Dev projects\crypto-trading-strategy-1\data\BTC_USDT_data.csv'
    main(filepath)