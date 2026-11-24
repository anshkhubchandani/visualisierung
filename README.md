# Improved parallel coordinates – interactive multivariate visualization (Python, Dash)

An interactive web app that makes **parallel coordinates plots** easier to read by
(1) **reordering the axes** automatically and (2) **colouring samples by k-means clusters** or by their
true class. Demonstrated on two classic multivariate datasets.

> University project (M.Sc. Bioinformatics, FSU Jena – module *Visualization*, team of 2, grade 1.0).

## Why
In a parallel coordinates plot the order of the axes decides which patterns you can see: correlations are
only visible between *neighbouring* axes. The default column order is arbitrary, so useful structure is
often hidden.

## What it does
- **Axis ordering** – computes the absolute Pearson correlation between all features and builds a greedy
  ordering from every possible start axis, keeping the best one:
  - **SIM**: neighbouring axes as similar as possible → smooth, readable bundles
  - **DIS**: neighbouring axes as dissimilar as possible → highlights contrasts
  - **Original**: column order of the dataset, for comparison
- **Clustering** – k-means (k = 3) on min-max-normalized features; switch the colouring between the
  true labels and the clusters to see how well unsupervised groups match the real classes
- **Interactive** – Plotly parallel coordinates with brushing on each axis; dropdowns switch ordering and
  colouring live

## Datasets
| Dataset | Samples | Features | Classes |
|---|---|---|---|
| UCI **Wine** | 178 | 13 chemical measurements | 3 cultivars |
| UCI **Auto MPG** (cars) | ~400 | 7 technical attributes | 3 origins |

## How to run
```bash
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:8050 in your browser.

## Project structure
```
app.py                    Dash app (layout + callbacks)
src/preprocessing.py      loading and min-max normalization
src/clustering.py         k-means clustering
src/ordering.py           similarity matrix and greedy SIM/DIS axis ordering
src/parallel_coordinates.py  plotting helpers
presentation/             data overview scripts and figures
wine_dataset/, cars_dataset/  data
```

## Tech
Python · pandas · NumPy · scikit-learn (k-means) · Plotly · Dash
