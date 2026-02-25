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
    SEX/GENDER : Uinvariate Analysis
    Interpretations : Clean, no missing values and it is a strong feature to consider
    """
    
    print(f"Count of each gender :: {df['Sex'].value_counts()}")    
    PRINT_SEPARATOR()

    sns.countplot(x='Sex', data=df)
    plt.title("Sex Distribution")
    plt.savefig("data/gender_dist")

if __name__ == "__main__":
    main()

