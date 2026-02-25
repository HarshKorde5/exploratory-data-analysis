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
    EMBARKED : Uinvariate Analysis
    Interpretations : One strong class, i.e. dominant port from where all boarded
                      Very less missing values(2).

                      Mode imputation will work, else dropping null values will also work.
    """
    print(f"Embarked distribution :: {df['Embarked'].value_counts(dropna=False)}")
    PRINT_SEPARATOR()

    sns.countplot(x='Embarked', data=df)
    plt.title("Embarked Countplot")
    plt.savefig('data/embarked_countplot.png')

if __name__ == "__main__":
    main()

