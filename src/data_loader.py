import pandas as pd


def load_wine_data():
    path = "wine_dataset/wine_table.csv"
    df = pd.read_csv(path)
    return df

def load_cars_data():
    path = "cars_dataset/auto-mpg.csv"
    df = pd.read_csv(path)
    return df

#for wine and cars

if __name__ == "__main__":
    wine_data = load_wine_data()
    cars_data = load_cars_data()

    print("Wine")
    print(wine_data.head())
    print(wine_data.shape)
    print(wine_data.columns)

    print("\nCars")
    print(cars_data.head())
    print(cars_data.shape)
    print(cars_data.columns)

# if __name__ == "__main__":
#     data = load_wine_data()
#     print(data.head())
#     print("\nShape:")
#     print(data.shape)
#     print("\nColumns:")
#     print(data.columns)
