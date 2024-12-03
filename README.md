# Crypto Trading Strategy 1 - Logistic Regression Model

This project implements a crypto trading strategy. The strategy is based on a logistic regression model, trained with technical indicators to predict whether the price of Bitcoin (BTC) will go up or down in the next period. 

## Key Features
- **Technical Indicators Used**:
  - **RSI (Relative Strength Index)**: Measures the speed and change of price movements.
  - **MACD (Moving Average Convergence Divergence)**: Indicates the strength of trends.
  - **VWAP (Volume Weighted Average Price)**: Reflects the average price of the asset based on volume.
  - **Volatility**: Historical volatility using log returns, helping to measure price fluctuations.

- **Model**: Logistic Regression to classify the price movement (up or down).
- **Optimization**: The model threshold has been adjusted to improve prediction performance.

## Getting Started  

### Prerequisites  
Install the required libraries:  
```bash
pip install -r requirements.txt
````

### Data
The project uses historical Bitcoin price data, specifically the following columns:

- close: Closing price of BTC/USDT.
- high: Highest price in a given period.
- low: Lowest price in a given period.
- volume: Trading volume.

### Usage
Modify the data source to use different cryptocurrency datasets.
Adjust the technical indicator windows or model parameters for optimization.
To improve prediction performance, explore different machine learning models (e.g., Random Forest, SVM).

### Contributing
Feel free to fork this repository, improve the code, and submit pull requests. If you have any questions or suggestions, please open an issue.

### License
This project is licensed under the Apache License - see the LICENSE file for details.

Have fun :).
