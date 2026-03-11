import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

#-------Data Loading--------#

df = pd.read_csv("data/dataset.csv")

#features
X = df.drop('median_house_value', axis=1)
#target
y = df['median_house_value']


#--------Data Observation & Preprocessing--------#
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

plt.figure(figsize=(8,5))
plt.scatter(df['median_income'], df['median_house_value'])
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.title('Median Income vs Median House Value')
plt.savefig('data/income_vs_price.png')


#--------Data Preprocessing--------#

# 1. Split data FIRST before calculating any stats
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Calculate median ONLY on training data to avoid data leakage
train_median = X_train['total_bedrooms'].median()

# 3. Fill missing values in both using training median
X_train = X_train.copy()
X_test = X_test.copy()
X_train['total_bedrooms'] = X_train['total_bedrooms'].fillna(train_median)
X_test['total_bedrooms'] = X_test['total_bedrooms'].fillna(train_median)

# 4. Encode categorical columns
X_train = pd.get_dummies(X_train, columns=['ocean_proximity'])
X_test = pd.get_dummies(X_test, columns=['ocean_proximity'])

# 5. Align columns (in case one set has categories the other doesn't)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)


#--------Model Training--------#
model = LinearRegression()
model.fit(X_train, y_train)


#--------Model Evaluation--------#
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")


#--------Model Prediction Graph--------#
# Take firt 50 samples for a clear comparison view
n = 50
actual_sample = y_test.values[:n]
predicted_sample = y_pred[:n]
x_indices = np.arange(n)

bar_width = 0.4
plt.figure(figsize=(18, 6))
plt.bar(x_indices - bar_width/2, actual_sample, width=bar_width, label='Actual', color='steelblue', alpha=0.8)
plt.bar(x_indices + bar_width/2, predicted_sample, width=bar_width, label='Predicted', color='tomato', alpha=0.8)

plt.xlabel('Sample Index')
plt.ylabel('House Price')
plt.title('Actual vs Predicted House Prices (First 50 Samples)')
plt.legend()
plt.tight_layout()
plt.savefig('data/actual_vs_predicted.png')