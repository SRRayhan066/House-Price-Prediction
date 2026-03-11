import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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