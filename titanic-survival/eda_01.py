import pandas as pd


def main():
    FILE_NAME = r'data/Titanic-Dataset.csv'
    df = pd.read_csv(FILE_NAME)

    print(df)

if __name__ == "__main__":
    main()

