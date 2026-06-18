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
    features = df.drop(columns=["name"], errors="ignore")
    # alles numerisch umwandeln
    features = features.apply(pd.to_numeric, errors="coerce")

    # model_year separat behalten
    model_year = features["model_year"]
    origin = features["origin"]
    # year und origin nicht skalieren (sollen fest bleiben)
    features_to_scale = features.drop(columns=["model_year", "origin"])

    # fehlende Werte entfernen
    features_to_scale = features_to_scale.dropna()
    target = target.loc[features_to_scale.index]
    model_year = model_year.loc[features_to_scale.index]
    origin = origin.loc[features_to_scale.index]

    # normalisieren
    normalized_features = (
        (features_to_scale - features_to_scale.min()) /
        (features_to_scale.max() - features_to_scale.min())
    )
    df_preprocessed = pd.concat([target, model_year, normalized_features], axis=1)
    return df_preprocessed

if __name__ == "__main__":
    wine_data = load_and_preprocess()
    print("Wine:")
    print(wine_data.head())
    print(wine_data.shape)

    cars_data = load_and_preprocess_cars()
    print("\nCars:")
    print(cars_data.head())
    print(cars_data.shape)


# if __name__ == "__main__":
#     data = load_and_preprocess_cars()
#     print(data.tail())
#     print(data.shape)