# Route-Level Delay Forecasting

Flight delay forecasting for the 50 busiest U.S. routes using XGBoost, LightGBM, LSTM, and TCN. Trained on 6 years of BTS flight records and Open-Meteo weather data.

**[Live dashboard](https://tisyasharma.github.io/flight-delay-forecasting/)** - explore the analysis and interact with model comparisons across routes and seasons.

[screenshot]

Gradient boosting models (XGBoost, LightGBM) hit ~11.8 min MAE and predict 77% of route-days within 15 minutes of actual. The deep learning models (LSTM, TCN) trail at ~13.5 min MAE since the 63 hand-engineered features already capture the temporal patterns they'd need to learn from scratch. Removing weather features bumps error up 10%, so weather data inclusion proves to be helpful.

The notebooks (`01-08`) walk through EDA, feature engineering, model training, and error analysis. Model results are also evaluated on the pages link. ()
