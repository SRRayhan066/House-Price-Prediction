import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """Load dataset from a CSV file and return the DataFrame."""
    df = pd.read_csv(file_path)
    return df


def observe_data(df):
    """Print key observations about the dataset."""
    print(df.head())
    print(df.tail())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.duplicated().sum())


def plot_income_vs_price(df, save_path='data/income_vs_price.png'):
    """Plot Median Income vs Median House Value and save the figure."""
    plt.figure(figsize=(8, 5))
    plt.scatter(df['median_income'], df['median_house_value'])
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value')
    plt.title('Median Income vs Median House Value')
    plt.savefig(save_path)
    plt.close()


def plot_correlation_matrix(df, save_path='data/correlation_matrix.png'):
    """Plot a heatmap of the correlation matrix for numeric columns."""
    plt.figure(figsize=(12, 8))
    # Filter for only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.savefig(save_path)
    plt.close()
