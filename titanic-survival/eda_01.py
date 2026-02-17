import pandas as pd

PRINT_SEPARATOR = lambda : print(100*'-')

def main():
    FILE_NAME = r'data/Titanic-Dataset.csv'
    df = pd.read_csv(FILE_NAME)

    PRINT_SEPARATOR()
    print("Titanic Survival Case Study")
    PRINT_SEPARATOR()

    print(df.head())

if __name__ == "__main__":
    main()

