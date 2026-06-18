from dash import Dash, dcc, html, Input, Output
import plotly.express as px

from src.preprocessing import load_and_preprocess, load_and_preprocess_cars
from src.clustering import add_kmeans_clusters
from src.ordering import compute_similarity_matrix, greedy_axis_order


# load data
wine_df = load_and_preprocess()
wine_df = add_kmeans_clusters(wine_df, target_column="class", n_clusters=3)

cars_df = load_and_preprocess_cars()
cars_df = add_kmeans_clusters(
    cars_df,
    target_column="origin",
    exclude_columns=["model_year"], #rausnehmen, weil ich model_year nicht normalisiert habe (fands komisch zu tun, weils ja jahreszahlen sind)
    n_clusters=3)

# feature columns
wine_target = "class"
cars_target = "origin"

wine_feature_columns = list(wine_df.drop(columns=[wine_target, "cluster"]).columns)
cars_feature_columns = list(cars_df.drop(columns=[cars_target, "cluster"]).columns)

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

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=color_title,
            tickvals=[1, 2, 3],
            ticktext=["1", "2", "3"]
        )
    )

    return fig


def create_parallel_coordinates_cars(ordering_name, color_by):
    selected_order = cars_axis_orders[ordering_name]
    dimensions = selected_order + [cars_target]

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
                tickvals=[1, 2, 3],
                ticktext=["USA", "Europe", "Japan"]
            ) if dim == cars_target else dict(
                label=dim,
                values=cars_df[dim]
            )
            for dim in dimensions
        ]
    )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title=color_title,
            tickvals=[1, 2, 3],
            ticktext=["1", "2", "3"]
        )
    )

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
        dcc.Graph(
            id="wine-parallel-coordinates-plot",
            figure=create_parallel_coordinates_wine("Original", "cluster")
        ),

        html.H2("Cars Dataset"),
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

    return wine_fig, cars_fig, descriptions[selected_ordering]

#run app
if __name__ == "__main__":
    app.run(debug=True)

# df = load_and_preprocess()
# df = add_kmeans_clusters(df, n_clusters=3)

# feature_columns = list(df.drop(columns=["class", "cluster"]).columns)

# similarity_matrix = compute_similarity_matrix(df, feature_columns)

# axis_orders = {
#     "Original": feature_columns,
#     "SIM": greedy_axis_order(similarity_matrix, mode="SIM"),
#     "DIS": greedy_axis_order(similarity_matrix, mode="DIS")
# }

# #checking ob SIM(DIS wwirklich andere reihnefolgen haben)
# print("\nOriginal order:")
# print(axis_orders["Original"])

# print("\nSIM order:")
# print(axis_orders["SIM"])

# print("\nDIS order:")
# print(axis_orders["DIS"])

# def create_parallel_coordinates(ordering_name, color_by):
#     selected_order = axis_orders[ordering_name]
#     dimensions = selected_order + ["class"]

#     color_title = "Original class" if color_by == "class" else "KMeans cluster"

#     fig = px.parallel_coordinates(
#         df,
#         dimensions=dimensions,
#         color=color_by,
#         range_color=[1, 3],
#         color_continuous_scale=[
#             "#f2d70b",
#             "#a56eff",
#             "#00CBC1"
#         ],
#         title=f"Wine Dataset - {ordering_name} axis order, colored by {color_title}"
#     )

#     fig.update_traces(
#         dimensions=[
#             dict(
#                 label=dim,
#                 values=df[dim],
#                 tickvals=[1, 2, 3],
#                 ticktext=["class 1", "class 2", "class 3"]
#             ) if dim == "class" else dict(
#                 label=dim,
#                 values=df[dim]
#             )
#             for dim in dimensions
#         ]
#     )

#     fig.update_layout(
#         coloraxis_colorbar=dict(
#             title=color_title,
#             tickvals=[1, 2, 3],
#             ticktext=["1", "2", "3"]
#         )
#     )

#     return fig

# #DASH / APP LAYOUT

# app = Dash(__name__)

# app.layout = html.Div(
#     children=[
#         html.H1("Improved Parallel Coordinates"),

#         html.P(
#             "Interactive Parallel Coordinates with KMeans clustering and axis reordering."
#         ),

#         html.Label("Axis ordering:"),

#         dcc.Dropdown(
#             id="ordering-dropdown",
#             options=[
#                 {"label": "Original order", "value": "Original"},
#                 {"label": "Similarity-based order (SIM)", "value": "SIM"},
#                 {"label": "Dissimilarity-based order (DIS)", "value": "DIS"},
#             ],
#             value="Original",
#             clearable=False
#         ),
#         html.Label("Color encoding:"),
#         dcc.Dropdown(
#             id="color-dropdown",
#             options=[
#                 {"label": "Original wine class", "value": "class"},
#                 {"label": "KMeans cluster", "value": "cluster"},
#                 ],
#                 value="cluster",
#                 clearable=False
#                 ),
        
#         # explaining the drop down options
#         html.Div(
#             id="ordering-description",
#             style={
#                 "marginTop": "15px",
#                 "marginBottom": "15px",
#                 "fontSize": "16px"
#             }
#         ),

#         dcc.Graph(
#             id="parallel-coordinates-plot",
#             figure=create_parallel_coordinates("Original", "cluster")
#         )
#     ]
# )


# @app.callback(
#     Output("parallel-coordinates-plot", "figure"),
#     Output("ordering-description", "children"),
#     Input("ordering-dropdown", "value"),
#     Input("color-dropdown", "value")
# )
# def update_plot(selected_ordering, selected_color):
#     descriptions = {
#         "Original": "Original order: The axes are shown in the order of the dataset.",
#         "SIM": "SIM order: Similar dimensions are placed next to each other.",
#         "DIS": "DIS order: Dissimilar dimensions are placed next to each other. This can make clusters more salient in cluttered data."
#     }

#     return create_parallel_coordinates(selected_ordering, selected_color), descriptions[selected_ordering]

# if __name__ == "__main__":
#     app.run(debug=True)