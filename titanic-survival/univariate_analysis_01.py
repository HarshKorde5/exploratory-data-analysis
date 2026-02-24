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
    Survived : Target variable UA(univariate analysis)
    Findings : 38% survived rate
    Slightly imbalanced, but not extreme
    """
    print(df['Survived'].value_counts())
    print(df['Survived'].value_counts(normalize=True))      #normalize means percentage

    PRINT_SEPARATOR()

    sns.countplot(x='Survived', data=df)
    plt.title("Survival Distribution")
    plt.savefig('data/survival_dist.png')

if __name__ == "__main__":
    main()

