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
    PCLASS : Uinvariate Analysis
    Interpretations : We can observe more number of passengers in lower class, this reflects real-world ship layout.
    """
    
    print(df['Pclass'].value_counts().sort_index())
    PRINT_SEPARATOR()
    
    sns.countplot(x='Pclass', data=df)
    plt.title("Pclass Countplot")
    plt.savefig('data/pclass_count.png')

if __name__ == "__main__":
    main()

