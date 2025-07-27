#1. Avg Close Price Per Year--------

import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("cleaned_tcs_stock.csv")

# Convert Date to datetime and extract year
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

# Group by year and calculate average Close price
yearly_avg = df.groupby('Year')['Close'].mean()

# Plot
plt.figure(figsize=(10, 5))
plt.plot(yearly_avg.index, yearly_avg.values, marker='o', color='orange')
plt.title('Average Close Price of TCS Per Year')
plt.xlabel('Year')
plt.ylabel('Average Close Price (INR)')
plt.grid(True)
plt.tight_layout()
plt.show()


#2.Close Price Over Time------------

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_tcs_stock.csv")
df['Date'] = pd.to_datetime(df['Date'])

plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Close'], color='blue')
plt.title('TCS Close Price Over Time')
plt.xlabel('Date')
plt.ylabel('Close Price (INR)')
plt.grid(True)
plt.tight_layout()
plt.show()


#3.Volume Traded Over Time------------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Step 1: Load cleaned data
df = pd.read_csv("cleaned_tcs_stock.csv")

# Step 2: Ensure 'Date' column is datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date'])

# Step 3: Sort by Date (just to be safe)
df = df.sort_values('Date')

# Step 4: Plot Volume vs Date
plt.figure(figsize=(12, 5))
plt.plot(df['Date'], df['Volume'], color='green')

# Step 5: Format X-axis to show only years
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Final chart formatting
plt.title('TCS Volume Traded Over Time (Year-wise)')
plt.xlabel('Year')
plt.ylabel('Volume')
plt.grid(True)
plt.tight_layout()
plt.xticks(rotation=45)
plt.show()

#4.Correlation Heatmap---------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the cleaned dataset
df = pd.read_csv('cleaned_tcs_stock.csv')

# Step 2: Create Correlation Matrix
corr = df.corr(numeric_only=True)

# Step 3: Plot the Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

#5.Scatter plot Close Price vs Volume----------

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("cleaned_tcs_stock.csv")
plt.figure(figsize=(8, 5))
plt.scatter(df['Close'], df['Volume'],  alpha=0.5, color='teal')
plt.title('Scatter Plot: Close Price vs Volume')
plt.xlabel('Close Price (INR)')
plt.ylabel('Volume')
plt.grid(True)
plt.tight_layout()
plt.show()


#6.TCS Stock Price Prediction (Date-wise)--------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 1: Load and clean data
df = pd.read_csv("cleaned_tcs_stock.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df[['Date', 'Close']].dropna()
df = df.sort_values('Date')
df.reset_index(drop=True, inplace=True)

# Step 2: Normalize Close prices
data = df[['Close']].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Step 3: Create sequences
X = []
y = []
sequence_len = 60

for i in range(sequence_len, len(scaled_data)):
    X.append(scaled_data[i-sequence_len:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Step 4: Split into train/test
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Step 5: Build and train model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Step 6: Predict & inverse scale
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 7: Plotting
test_dates = df['Date'].iloc[-len(actual_prices):]

plt.figure(figsize=(14, 5))
plt.plot(test_dates, actual_prices, label='Actual Price', color='blue', linewidth=1.5)
plt.plot(test_dates, predicted_prices, label='Predicted Price', color='red', linewidth=1.5)
plt.title('TCS Stock Price Prediction (Date-wise)')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




#7.Summary Table----------------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned TCS data
df = pd.read_csv('cleaned_tcs_stock.csv')

# Plot heatmap of summary statistics
plt.figure(figsize=(10, 1))
sns.heatmap(df.describe().T[['mean', 'std', 'min', 'max']],
             annot=True, fmt='.2f', cmap='YlGnBu')
plt.title("Summary Stats of TCS Data")
plt.tight_layout()
plt.show()
