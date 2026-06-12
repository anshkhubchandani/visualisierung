import pandas as pd


def load_wine_data():

    path = "wine_dataset/wine_table.csv"

    df = pd.read_csv(path)

    return df


if __name__ == "__main__":

    data = load_wine_data()

    print(data.head())

    print("\nShape:")
    print(data.shape)

    print("\nColumns:")
    print(data.columns)
