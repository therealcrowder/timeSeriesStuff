# Packages
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

# Bring data in
# Use your own path
df = pd.read_excel("Superstore.xls")
furniture = df.loc[df['Category'] == 'Furniture']

# Check first few rows
# Get the first few rows
furniture.head()
# 21 columns

# Min and Max dates
furniture['Order Date'].min(), furniture['Order Date'].max()
# 4 years worth of data from 2014 through 2017

# Remove uneeded columns and general cleaning

cols = ['Row ID', 'Order ID', 'Ship Date', 'Ship Mode', 
        'Customer ID', 'Customer Name', 'Segment', 'Country', 
        'City', 'State', 'Postal Code', 'Region', 'Product ID', 
        'Category', 'Sub-Category', 'Product Name', 'Quantity', 
        'Discount', 'Profit']

furniture.drop(cols, axis=1, inplace=True)
furniture = furniture.sort_values('Order Date')
# Time and Sales
furniture.isnull().sum()

# Index
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()

# Set index 
furniture = furniture.set_index('Order Date')
furniture.index

y = furniture['Sales'].resample('MS').mean()

# Plot
y.plot(figsize = (15, 6))
plt.show()

# Decomposition - Trend - Seasonality - Noise
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8

decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

# ARIMA Examples
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) 
                for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# Grid Search | Parameter Selection
# Watch spacing
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y, 
                                            order=param,seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

# The above output would lead us to select the 
# SARIMAX(1, 1, 1)x(1, 1, 0, 12) based on the AIC value of 297.78

# Fitting the model

mod = sm.tsa.statespace.SARIMAX(y,
                               order = (1, 1, 1),
                               seasonal_order = (1, 1, 0, 12),
                               enforce_stationarity = False,
                               enforce_invertibility = False)
results = mod.fit()

print(results.summary().tables[1])

# Model Plots
results.plot_diagnostics(figsize=(16, 8))
plt.show()

# Validation
pred = results.get_prediction(start = pd.to_datetime('2017-01-01'),
                             dynamic = False)
pred_ci = pred.conf_int()

ax = y['2014':].plot(label = 'observed')
pred.predicted_mean.plot(ax = ax, label = 'One-step Ahead Forecast',
                        alpha = .7, figsize = (14, 7))

ax.fill_between(pred_ci.index,
               pred_ci.iloc[:, 0],
               pred_ci.iloc[:, 1], color = 'k', alpha = .2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()

plt.show()

# RMSE
y_forecasted = pred.predicted_mean
y_truth = y['2017-01-01':]

mse = ((y_forecasted - y_truth) ** 2).mean()
print('The MSE of our forecast is {}'.format(round(mse, 2)))

print('The RMSE of our forecast is {}'.format(round(np.sqrt(mse), 2)))

# Longer Horizon with Furniture
pred_uc = results.get_forecast(steps = 100) # going out 100 months
pred_ci = pred_uc.conf_int()

ax = y.plot(label = 'Observed', figsize = (14, 7))
pred_uc.predicted_mean.plot(ax = ax, label = 'Forecast')
ax.fill_between(pred_ci.index,
               pred_ci.iloc[:, 0],
               pred_ci.iloc[:, 1], color = 'k', alpha = .25)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')

plt.legend()
plt.show()

# Office Supplies and furniture

furniture = furniture.sort_values('Order Date')
office = office.sort_values('Order Date')

furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
office = office.groupby('Order Date')['Sales'].sum().reset_index()

furniture = furniture.set_index('Order Date')
office = office.set_index('Order Date')

# Get monthly data
y_furniture = furniture['Sales'].resample('MS').mean()
y_office = office['Sales'].resample('MS').mean()

furniture = pd.DataFrame({'Order Date':y_furniture.index, 'Sales':y_furniture.values})
office = pd.DataFrame({'Order Date': y_office.index, 'Sales': y_office.values})

# Combine the sets
store = furniture.merge(office, how='inner', on='Order Date')
store.rename(columns={'Sales_x': 'furniture_sales', 'Sales_y': 'office_sales'}, inplace=True)

store.head()

# Plots

plt.figure(figsize=(20, 8))

plt.plot(store['Order Date'], 
         store['furniture_sales'], 
         'b-', label = 'Furniture')
plt.plot(store['Order Date'], 
         store['office_sales'], 
         'r-', label = 'Office Supplies')
plt.xlabel('Date'); plt.ylabel('Sales'); plt.title('Sales of Furniture and Office Supplies')
plt.legend();
plt.legend()
plt.show()

#### Prophet ####

from fbprophet import Prophet

# Furniture Model Prep
furniture = furniture.rename(columns = 
                             {'Order Date': 'ds', 'Sales': 'y'})
furniture_model = Prophet(interval_width = 0.95)
furniture_model.fit(furniture)

# Office Supplies Model Prep
office = office.rename(columns = 
                      {'Order Date': 'ds', 'Sales': 'y'})
office_model = Prophet(interval_width = 0.95)
office_model.fit(office)

# Furniture Forecast
furniture_forecast = furniture_model.make_future_dataframe(
periods = 36, freq = 'MS')
furniture_forecast = furniture_model.predict(furniture_forecast)

# Office Supplies Forecast
office_forecast = office_model.make_future_dataframe(
periods = 36, freq = 'MS')
office_forecast = office_model.predict(office_forecast)

# Plot Furniture
plt.figure(figsize = (18, 6))
furniture_model.plot(furniture_forecast, xlabel = 'Date',
                     ylabel = 'Sales')
plt.title('Furniture Sales');
plt.show()

# Plot Office Supplies
plt.figure(figsize = (18, 6))
furniture_model.plot(office_forecast, xlabel = 'Date',
                     ylabel = 'Sales')
plt.title('Office Supply Sales');
plt.show()

# Data frame of forecast data
furniture_names = ['furniture_%s' % column for column in furniture_forecast.columns]

office_names = ['office_%s' % column for column in office_forecast.columns]

merge_furniture_forecast = furniture_forecast.copy()
merge_office_forecast = office_forecast.copy()
merge_furniture_forecast.columns = furniture_names
merge_office_forecast.columns = office_names

forecast = pd.merge(merge_furniture_forecast,
                    merge_office_forecast, how = 'inner',
                    left_on = 'furniture_ds', right_on = 'office_ds')
forecast = forecast.rename(columns={'furniture_ds': 'Date'}).drop('office_ds', axis=1)
forecast.head()

# Trend plot
plt.figure(figsize=(10, 7))

plt.plot(forecast['Date'], forecast['furniture_trend'], 'b-')
plt.plot(forecast['Date'], forecast['office_trend'], 'r-')

plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')
plt.title('Furniture vs. Office Supplies Sales Trend');

plt.show()

# Plots of both

plt.figure(figsize=(10, 7))

plt.plot(forecast['Date'], forecast['furniture_yhat'], 'b-')
plt.plot(forecast['Date'], forecast['office_yhat'], 'r-')

plt.legend(); plt.xlabel('Date'); plt.ylabel('Sales')
plt.title('Furniture vs. Office Supplies Estimate');
plt.show()

# Trends by product
# Furniture
furniture_model.plot_components(furniture_forecast);
plt.show()

office_model.plot_components(office_forecast);
plt.show()