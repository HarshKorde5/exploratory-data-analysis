import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PRINT_SEPARATOR = lambda : print(100*'-')

sns.set_theme(style='whitegrid')


def main():
    FILE_NAME = r'data/Titanic-Dataset.csv'
    df = pd.read_csv(FILE_NAME)

    PRINT_SEPARATOR()
    print("Titanic Survival Case Study")
    PRINT_SEPARATOR()
    
    """
    CABIN : Uinvariate Analysis
    Interpretations : Huge amount of null values. 
                        Total 77% of missing values.

                        Too many unique values present, data too sparse and noisy

                        Best way is to drop this feature

    """
    
    print(f"CABIN : Count of null values :: {df['Cabin'].isnull().sum()}")
    print(f"Total percentage :: {(df['Cabin'].isnull().mean() * 100).round(2)}%")

    print(f" Count of Unqiue values/categories :: {df['Cabin'].nunique()}")


if __name__ == "__main__":
    main()

