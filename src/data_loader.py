import pandas as pd
import matplotlib.pyplot as plt


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
