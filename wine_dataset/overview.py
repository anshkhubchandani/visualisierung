import pandas as pd
import matplotlib.pyplot as plt

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

df = pd.read_csv("/Users/anshulkhubchandani/Documents/3Semester/Visualisierung/visualisierung/wine_dataset/wine.data", header=None, names=columns)

print(df.head())
print(df.info())
print(df.describe())
####DESCRIBE##### -> Describe also includes class but class is not a real numeric feature!
features = df.drop(columns="class")

print(features.describe())

print(df["class"].value_counts().sort_index())

#table
#df.to_csv("wine_table.csv", index=False)

#schön für die Präsentation: Balkendiagramm der Anzahl der Samples pro Klasse

class_counts = df["class"].value_counts().sort_index()
class_counts.plot( kind="bar", color=["#B4CEF7", "#CDEFD5", "#EDC4C5"],rot=0)

plt.xlabel("Wine class")
plt.ylabel("Number of samples")
plt.title("Number of wine samples per class")
plt.tight_layout()

plt.show()