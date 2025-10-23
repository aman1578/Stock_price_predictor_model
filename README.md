Stock Price Predictor

A Streamlit app that trains a Prophet time-series model (with simple feature regressors and a linear-correction step) to predict stock closing prices from a CSV/TXT containing date and price columns.

Features

1. Load CSV / TXT stock data via sidebar.

2. Select date and price columns interactively.

3. Feature engineering: day of week, month.

4. Train/test split (70/30) and Prophet model with weekly & yearly seasonality.

5. Optional bias correction using a simple Linear Regression on test predictions.

6. Visuals using Plotly: train / actual / forecast.

7. Single-date future prediction UI.
