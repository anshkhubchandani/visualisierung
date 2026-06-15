#test -> kann gelöscht werden
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from preprocessing import df_preprocessed

plt.figure(figsize=(14, 7))
parallel_coordinates(df_preprocessed, "class", colormap="viridis", alpha=0.4)

plt.title("Parallel Coordinates Plot - Wine Dataset")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()