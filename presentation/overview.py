#anshul
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("wine_dataset/wine_table.csv")

print(df.head())
print(df.info())
print(df.describe())
####DESCRIBE##### -> Describe also includes class but class is not a real numeric feature!
features = df.drop(columns="class")
print(features.describe())

print(df["class"].value_counts().sort_index())



#schön für die Präsentation: Balkendiagramm der Anzahl der Samples pro Klasse

class_counts = df["class"].value_counts().sort_index()
class_counts.plot( kind="bar", color=["#f2d70b", "#a56eff", "#009d9a"],rot=0)


plt.xlabel("Wine class")
plt.ylabel("Number of samples")
plt.title("Number of wine samples per class")
plt.tight_layout()

plt.show()