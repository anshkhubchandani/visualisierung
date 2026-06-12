
import pandas as pd
from dash import Dash, dcc, html
import plotly.express as px
from src.preprocessing import load_and_preprocess
from src.preprocessing import load_and_preprocess
from src.clustering import add_kmeans_clusters
""" 
Baseline: 
wir können spätwer zeigen normaler reihenfolge -> SIM -> DIS
"""

df = load_and_preprocess()
features = df.drop(columns=["class"]).columns


#wir normalisieren hier nur features und nicht class (class is label)
# features = all columns except class (nicht numerical)

# Create Parallel Coordinates
fig = px.parallel_coordinates(
    df,
    dimensions=features,
    color="class",
    color_continuous_scale=[ #damit es endlich kategorisch aussieht!!!
        "#B4CEF7",
        "#CDEFD5",
        "#EDC4C5"
    ],
    title="Wine Dataset - Parallel Coordinates"
)

#plotly nimmt color= class als kontinuierliche num variable!!! also zeitskala s
#hier versuch ich sie zu verstecken!
fig.update_layout(
    coloraxis_colorbar=dict(
        title="class",
        tickvals=[1, 2, 3],
        ticktext=["1", "2", "3"]
    )
)

# Dash App buttonssss
app = Dash(__name__)
app.layout = html.Div(

    children=[
        html.H1(
            "Improved Parallel Coordinates"
        ),
        html.P(
            "Baseline visualization of the Wine dataset"
        ),
        dcc.Graph(
            figure=fig
        )
    ]

)



if __name__ == "__main__":
    app.run(debug=True)