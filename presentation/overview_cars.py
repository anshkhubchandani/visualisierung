import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("cars_dataset/auto-mpg.csv")

print(df.head())
print(df.info())
print(df.describe())

features = df.drop(columns="origin")
print(features.describe())

print(df["origin"].value_counts().sort_index())

class_counts = df["origin"].value_counts().sort_index()
class_counts.plot( kind="bar", color=["#f2d70b", "#a56eff", "#009d9a"],rot=0)


plt.xlabel("Cars class")
plt.ylabel("Number of samples")
plt.title("Number of cars samples per origin (class)")
plt.tight_layout()

plt.show()