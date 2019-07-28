# ENSEMBLE-TIME-SERIES-MODELING-
Stacking based ensemble time series model , with 2 layers.
1st layer learning - try to learn the existing forecasting along with event calender flags with a stack based approach.
2nd Layer - Model the Error while doing the forecasting
Base learners inputs - ARIMA_FORECAST,LINER_REGRESSION_FORECAST,ETS_FORECAST,LSTM_FORECAST,MOVING_AVERGAE_FORECAST & EVENT_CALENDER
Meta learners inputs - Error & Forecasted 
