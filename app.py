from dash import Dash, dcc, html, Input, Output
import plotly.express as px
from src.preprocessing import load_and_preprocess, load_and_preprocess_cars
from src.clustering import add_kmeans_clusters
from src.ordering import compute_similarity_matrix, greedy_axis_order


# load data
wine_df = load_and_preprocess()
wine_df = add_kmeans_clusters(wine_df, target_column="class", n_clusters=3)

cars_df, cars_original_df = load_and_preprocess_cars()
cars_df = add_kmeans_clusters(
    cars_df,
    target_column="origin",
    exclude_columns=["model_year"], #rausnehmen, weil ich model_year nicht normalisiert habe (fands komisch zu tun, weils ja jahreszahlen sind)
    n_clusters=3)

# feature columns
wine_target = "class"
cars_target = "origin"

wine_feature_columns = list(wine_df.drop(columns=[wine_target, "cluster"]).columns)
cars_feature_columns = list(cars_df.drop(columns=[cars_target, "cluster", "model_year"]).columns)

# similarity matrices
wine_similarity_matrix = compute_similarity_matrix(wine_df, wine_feature_columns)
cars_similarity_matrix = compute_similarity_matrix(cars_df, cars_feature_columns)

# axis orders
wine_axis_orders = {
    "Original": wine_feature_columns,
    "SIM": greedy_axis_order(wine_similarity_matrix, mode="SIM"),
    "DIS": greedy_axis_order(wine_similarity_matrix, mode="DIS")
}

cars_axis_orders = {
    "Original": cars_feature_columns,
    "SIM": greedy_axis_order(cars_similarity_matrix, mode="SIM"),
    "DIS": greedy_axis_order(cars_similarity_matrix, mode="DIS")
}

# debug prints
print("\nWine - Original order:")
print(wine_axis_orders["Original"])
print("\nWine - SIM order:")
print(wine_axis_orders["SIM"])
print("\nWine - DIS order:")
print(wine_axis_orders["DIS"])

print("\nCars - Original order:")
print(cars_axis_orders["Original"])
print("\nCars - SIM order:")
print(cars_axis_orders["SIM"])
print("\nCars - DIS order:")
print(cars_axis_orders["DIS"])


# figure functions
def create_parallel_coordinates_wine(ordering_name, color_by):
    selected_order = wine_axis_orders[ordering_name]
    dimensions = selected_order + [wine_target]

    color_title = "Original class" if color_by == wine_target else "KMeans cluster"

    fig = px.parallel_coordinates(
        wine_df,
        dimensions=dimensions,
        color=color_by,
        range_color=[1, 3],
        color_continuous_scale=["#f2d70b", "#a56eff", "#00CBC1"],
        title=f"Wine Dataset - {ordering_name} axis order, colored by {color_title}"
    )

    fig.update_traces(
        dimensions=[
            dict(
                label=dim,
                values=wine_df[dim],
                tickvals=[1, 2, 3],
                ticktext=["class 1", "class 2", "class 3"]
            ) if dim == wine_target else dict(
                label=dim,
                values=wine_df[dim]
            )
            for dim in dimensions
        ]
    )

    fig.update_layout(coloraxis_showscale=False)
    return fig


# new for original axis values (cars)!!!
def create_axis_with_original_ticks(dim, normalized_df, original_df, n_ticks=5):
    orig_min = original_df[dim].min()
    orig_max = original_df[dim].max()

    if orig_min == orig_max:
        return dict(
            label=dim,
            values=normalized_df[dim]
        )

    ticktext_vals = [
        orig_min + (orig_max - orig_min) * i / (n_ticks - 1)
        for i in range(n_ticks)
    ]
    
    # originalwerte in normierte tick-position umrechnen
    if dim == "model_year":
        ticktext_vals = [round(v) for v in ticktext_vals]
        ticktext_vals = sorted(list(dict.fromkeys(ticktext_vals)))
    else:
        ticktext_vals = [round(v, 1) for v in ticktext_vals]

    tickvals_norm = [
        (v - orig_min) / (orig_max - orig_min)
        for v in ticktext_vals
    ]

    return dict(
        label=dim,
        values=normalized_df[dim],
        tickvals=tickvals_norm,
        ticktext=[str(v) for v in ticktext_vals]
    )


def create_parallel_coordinates_cars(ordering_name, color_by):
    selected_order = cars_axis_orders[ordering_name]
    dimensions = selected_order + ["model_year", cars_target]

    color_title = "Original origin" if color_by == cars_target else "KMeans cluster"

    fig = px.parallel_coordinates(
        cars_df,
        dimensions=dimensions,
        color=color_by,
        range_color=[1, 3],
        color_continuous_scale=["#f2d70b", "#a56eff", "#00CBC1"],
        title=f"Cars Dataset - {ordering_name} axis order, colored by {color_title}"
    )

    fig.update_traces(
        dimensions=[
            dict(
                label=dim,
                values=cars_df[dim],
                tickvals=[70, 73, 76, 79, 82],
                ticktext=["1970", "1973","1976", "1979","1982"]
            ) if dim == "model_year" else dict(
                label=dim,
                values=cars_df[dim],
                tickvals=[1, 2, 3],
                ticktext=["USA", "Europe", "Japan"]
            ) if dim == cars_target else create_axis_with_original_ticks(
                dim,
                cars_df,
                cars_original_df
            )
            for dim in dimensions
        ]
    )

    fig.update_layout(coloraxis_showscale=False)
    return fig

# app layout
app = Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1("Improved Parallel Coordinates"),

        html.P(
            "Interactive Parallel Coordinates with KMeans clustering and axis reordering for two datasets."
        ),

        html.Label("Axis ordering:"),
        dcc.Dropdown(
            id="ordering-dropdown",
            options=[
                {"label": "Original order", "value": "Original"},
                {"label": "Similarity-based order (SIM)", "value": "SIM"},
                {"label": "Dissimilarity-based order (DIS)", "value": "DIS"},
            ],
            value="Original",
            clearable=False
        ),

        html.Label("Color encoding:"),
        dcc.Dropdown(
            id="color-dropdown",
            options=[
                {"label": "Original class / origin", "value": "target"},
                {"label": "KMeans cluster", "value": "cluster"},
            ],
            value="cluster",
            clearable=False
        ),

        html.Div(
            id="ordering-description",
            style={
                "marginTop": "15px",
                "marginBottom": "15px",
                "fontSize": "16px"
            }
        ),

        html.H2("Wine Dataset"), 
        html.Div(
            id="wine-legend",
            style={"fontSize": "18px", "marginBottom": "10px"}
        ),


        dcc.Graph(
            id="wine-parallel-coordinates-plot",
            figure=create_parallel_coordinates_wine("Original", "cluster")
        ),

        
        html.H2("Cars Dataset"),
        html.Div(
            id="cars-legend",
            style={ "fontSize": "18px", "marginBottom": "10px"}
        ),
        
        dcc.Graph(
            id="cars-parallel-coordinates-plot",
            figure=create_parallel_coordinates_cars("Original", "cluster")
        )
    ],
    style={"padding": "20px"}
)

# callback
@app.callback(
    Output("wine-parallel-coordinates-plot", "figure"),
    Output("cars-parallel-coordinates-plot", "figure"),
    Output("ordering-description", "children"),
    Output("wine-legend", "children"),
    Output("cars-legend", "children"),
    Input("ordering-dropdown", "value"),
    Input("color-dropdown", "value")
)

def update_plots(selected_ordering, selected_color):
    descriptions = {
        "Original": "Original order: The axes are shown in the order of the dataset.",
        "SIM": "SIM order: Similar dimensions are placed next to each other.",
        "DIS": "DIS order: Dissimilar dimensions are placed next to each other. This can make clusters more salient in cluttered data."
    }

    wine_color = wine_target if selected_color == "target" else "cluster"
    cars_color = cars_target if selected_color == "target" else "cluster"

    wine_fig = create_parallel_coordinates_wine(selected_ordering, wine_color)
    cars_fig = create_parallel_coordinates_cars(selected_ordering, cars_color)

    
    if selected_color == "target":
        wine_legend = [
        html.Span("■ Wine class 1 ", style={"color": "#f2d70b", "marginRight": "20px"}),
        html.Span("■ Wine class 2 ", style={"color": "#a56eff", "marginRight": "20px"}),
        html.Span("■ Wine class 3 ", style={"color": "#00CBC1"}),
    ]
    else:
        wine_legend = [
            html.Span("■ KMeans cluster 1 ", style={"color": "#f2d70b", "marginRight": "20px"}),
            html.Span("■ KMeans cluster 2 ", style={"color": "#a56eff", "marginRight": "20px"}),
            html.Span("■ KMeans cluster 3 ", style={"color": "#00CBC1"}),
            ]
        
    if selected_color == "target":
        cars_legend = [ 
            html.Span( "■ USA ", style={"color": "#f2d70b", "marginRight": "20px"}),
            html.Span("■ Europe ", style={"color": "#a56eff", "marginRight": "20px"}),
            html.Span("■ Japan ", style={"color": "#00CBC1"}),
            ]
    else: 
        cars_legend = [
            html.Span("■ KMeans cluster 1 ", style={"color": "#f2d70b", "marginRight": "20px"}),
            html.Span("■ KMeans cluster 2 ", style={"color": "#a56eff", "marginRight": "20px"}),
            html.Span("■ KMeans cluster 3 ", style={"color": "#00CBC1"}),
            ]
    return wine_fig, cars_fig, descriptions[selected_ordering], wine_legend, cars_legend
    

#run app
if __name__ == "__main__":
    app.run(debug=True)