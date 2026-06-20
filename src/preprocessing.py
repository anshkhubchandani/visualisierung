from pathlib import Path
import pandas as pd

# wine dataset
def load_and_preprocess():
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / "wine_dataset" / "wine_table.csv"
    df = pd.read_csv(file_path)


    # separate features and class label
    features = df.drop(columns=["class"])
    target = df["class"]

    

    # Scaling and normalization -> important to make data comparable in plot
    # ## Normalization so that values are between 0 and 1

    # min-max normalization    
    normalized_features = (
        (features - features.min()) / (features.max() - features.min())
    )


    # combine class + normalized features again
    df_preprocessed = pd.concat(
        [target, normalized_features],
        axis=1
    )

    return df_preprocessed


# only for testing this file
if __name__ == "__main__":
    data = load_and_preprocess()
    print(data.head())
    print(data.shape)


# cars dataset
def load_and_preprocess_cars():
    base_dir = Path(__file__).resolve().parent.parent
    file_path = base_dir / "cars_dataset" / "auto-mpg.csv"
    df = pd.read_csv(file_path)

    target = df["origin"]
    features = df.drop(columns=["name"], errors="ignore") #we took all features except name, which is not numeric (maybe we should have dropped model_year as well?)

    #remove non-numeric name column
    df = df.drop(columns=["name"], errors="ignore")
    # Missing-Value-Zeichen im Auto-Datensatz abfangen
    df = df.replace("?", pd.NA)
    # alles numerisch umwandeln
    df = df.apply(pd.to_numeric, errors="coerce")
    #remove missing vlaues
    df = df.dropna().reset_index(drop=True)

    # originaldaten für achsenbeschriftung
    df_original = df.copy()

    # origin nicht skalieren
    target = df["origin"]
    # zeige model_year, wird aber nicht benutzt für clustering /order
    model_year = df["model_year"]
    
    features_to_scale = df.drop(columns=["origin", "model_year"])                                         

    # normalisieren
    normalized_features = (
        (features_to_scale - features_to_scale.min()) /
        (features_to_scale.max() - features_to_scale.min())
    )

    df_preprocessed = pd.concat([target, normalized_features,model_year], axis=1)
    return df_preprocessed, df_original

if __name__ == "__main__":
    wine_data = load_and_preprocess()
    print("Wine:")
    print(wine_data.head())
    print(wine_data.shape)

      # cars
    cars_data, cars_original_data = load_and_preprocess_cars()
    print("\nCars original:")
    print(cars_original_data.head())
    print("\nCars normalized:")
    print(cars_data.head())


# if __name__ == "__main__":
#     data = load_and_preprocess_cars()
#     print(data.tail())
#     print(data.shape)