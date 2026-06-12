from pathlib import Path
import pandas as pd


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