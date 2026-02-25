import numpy as np
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
    FARE : Uinvariate Analysis
    Interpretations : 
        mean is greater than the median
        observe the min and max value (0, 512) which is huge skew
        It is a positive skew with heavy right tail, meaning many passangers purchased lower priced tickets,while a few purchased high priced tickets

        Log transformation might be used to bring the values to a managable scale.

        The box plot shows many outliers, but as fare reflects the wealth of passenger, we cannot drop it.

    """

    print(df['Fare'].describe())
    PRINT_SEPARATOR()

    sns.histplot(df['Fare'], bins=40, kde=True)
    plt.title("Fare distribution")
    plt.savefig("data/fare_dist.png")
    plt.close()

    sns.histplot(np.log1p(df['Fare']), bins=40)
    plt.title("Log Fare distribution")
    plt.savefig("data/log_fare_dist.png")

    sns.boxplot(x = df['Fare'])
    plt.title("Fare box plot")
    plt.savefig("data/fare_boxplot.png")



if __name__ == "__main__":
    main()

