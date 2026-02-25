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
    Age : UA(univariate analysis)
    Findings : Count of null values is 177
    Concerning factor as age is imp feature for survival model
    Min ~ 0.42 (infants)
    Max ~ 80 (reasonable)
    Mean > Median -> right skew

    Multiple peaks (children + adults)
    Median imputation preferred over mean

    Older passengers flagged as outliers
    But real-world it is always valid
    So no need to handle/remove outliers
    """

    print(f"Count of null values in AGE ::  {df['Age'].isnull().sum()}")
    PRINT_SEPARATOR()
    print(f"Summary Statistics :: {df['Age'].describe()}")

    PRINT_SEPARATOR()

    sns.histplot(df['Age'], bins=30, kde=True)
    plt.title("Age Distribution")
    plt.savefig('data/age_dist.png')

    sns.boxplot(x = df['Age'])
    plt.title("Age Boxplot")
    plt.savefig('data/age_boxplot.png')

    print(f"Skewness of age :: {df['Age'].skew()}")
    PRINT_SEPARATOR()

    """
    Age imputations methods
    Using mean will pull to outliers as we saw it is right skewed.
    Use of median:  
    1 : global median of age column
    2 : group wise median of age column, 
        example : median age per passenger class, because higher class (1) were old passangers compared to class 2,3 which had younger and families
    """

    #method 1 : global median
    # age_median = df['Age'].median()
    # df['Age'] = df['Age'].fillna(age_median)


    #method 2 : group-wise median
    df['Age'] = df.groupby('Pclass')['Age'].transform(
        lambda x : x.fillna(x.median())
    )


    print(f"Count of null values in AGE column after median imputation :: {df['Age'].isna().sum()}")


if __name__ == "__main__":
    main()

