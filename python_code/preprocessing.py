from pathlib import Path
import pandas as pd

# Get data using relative paths

base_dir = Path(__file__).resolve().parent.parent
file_path = base_dir / "wine_dataset" / "wine.data"

columns = [
    "class",
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline"
]

df = pd.read_csv(file_path, header=None, names=columns)

# Overview about the data

print("\nOriginal Data:")
print(df.head())
print(df.shape) # gives rows and columns
#print(df["class"].value_counts())

features = df.drop(columns=["class"])
target = df["class"]

# Scaling and normalization -> important to make data comparable in plot
## Normalization so that values are between 0 and 1

normalized_features = (features - features.min()) / (features.max() - features.min())

df_preprocessed = pd.concat([target, normalized_features], axis=1)

print("\nPreprocessed Data:")
print(df_preprocessed.head())
print(df_preprocessed.shape)
