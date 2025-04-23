import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Load the data
df = pd.read_excel("CocaCola_Sales_Rawdata.xlsx", sheet_name='Sheet1')

# Convert 'Quarter' to datetime
quarter_to_month = {'Q1': '01', 'Q2': '04', 'Q3': '07', 'Q4': '10'}
df['Year'] = df['Quarter'].apply(lambda x: '19' + x.split('_')[1])
df['Month'] = df['Quarter'].apply(lambda x: quarter_to_month[x.split('_')[0]])
df['Date'] = pd.to_datetime(df['Year'] + '-' + df['Month'])
df.set_index('Date', inplace=True)

# Plot the sales
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='Sales')
plt.title('CocaCola Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# Check for stationarity
adf_result = adfuller(df['Sales'])
print("ADF Test p-value:", adf_result[1])

# Fit ARIMA model
model = ARIMA(df['Sales'], order=(1, 1, 1))  # You can tweak (p,d,q) based on diagnostics
model_fit = model.fit()
print(model_fit.summary())

# Forecast next 8 quarters
forecast_steps = 8
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=3), periods=forecast_steps, freq='Q')
forecast_df = pd.DataFrame({
    'Forecast': forecast.predicted_mean,
    'Lower CI': forecast.conf_int().iloc[:, 0],
    'Upper CI': forecast.conf_int().iloc[:, 1]
}, index=forecast_index)

# Plot forecast
plt.figure(figsize=(12, 6))
plt.plot(df['Sales'], label='Historical Sales')
plt.plot(forecast_df['Forecast'], label='Forecast', color='orange')
plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='orange', alpha=0.3)
plt.title('Sales Forecast using ARIMA')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()