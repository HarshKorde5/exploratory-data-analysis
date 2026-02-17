import pandas as pd

PRINT_SEPARATOR = lambda : print(100*'-')

def main():
    FILE_NAME = r'data/Titanic-Dataset.csv'
    df = pd.read_csv(FILE_NAME)

    PRINT_SEPARATOR()
    print("Titanic Survival Case Study")


    PRINT_SEPARATOR()
    print("Data info")
    PRINT_SEPARATOR()
    print(df.info())

    PRINT_SEPARATOR()
    print("Duplicate entries")
    PRINT_SEPARATOR()
    print(df[df.duplicated()])
    
    PRINT_SEPARATOR()
    print("Data describe")
    PRINT_SEPARATOR()

    print(df.describe())

    PRINT_SEPARATOR()
    print("Missing values")
    PRINT_SEPARATOR()
    print(df.isnull().sum())

    print(f"Total missing values cells in dataset : {df.isnull().sum().sum()}")
    print(f"Missing values percentage is : {round(df.isna().sum().sum() / df.size*100, 2)}%")




if __name__ == "__main__":
    main()

