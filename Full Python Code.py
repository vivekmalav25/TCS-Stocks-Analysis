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


#6.Summary Table----------------

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
