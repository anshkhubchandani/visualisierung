from dash import Dash, dcc, html, Input, Output
import plotly.express as px

from src.preprocessing import load_and_preprocess
from src.clustering import add_kmeans_clusters
from src.ordering import compute_similarity_matrix, greedy_axis_order


df = load_and_preprocess()
df = add_kmeans_clusters(df, n_clusters=3)

feature_columns = list(df.drop(columns=["class", "cluster"]).columns)

similarity_matrix = compute_similarity_matrix(df, feature_columns)
print(similarity_matrix)

axis_orders = {
    "Original": feature_columns,
    "SIM": greedy_axis_order(similarity_matrix, mode="SIM"),
    "DIS": greedy_axis_order(similarity_matrix, mode="DIS")
}

#checking ob SIM(DIS wwirklich andere reihnefolgen haben)
print("\nOriginal order:")
print(axis_orders["Original"])

print("\nSIM order:")
print(axis_orders["SIM"])

print("\nDIS order:")
print(axis_orders["DIS"])

from src.ordering import print_axis_order_debug

print_axis_order_debug(similarity_matrix, axis_orders["SIM"], "SIM")
print_axis_order_debug(similarity_matrix, axis_orders["DIS"], "DIS")


#legend verstecken 
def create_parallel_coordinates(ordering_name, color_by):
    selected_order = axis_orders[ordering_name]
    dimensions = selected_order + ["class"]

    color_title = "Original class" if color_by == "class" else "KMeans cluster"

    fig = px.parallel_coordinates(
        df,
        dimensions=dimensions,
        color=color_by,
        range_color=[1, 3],
        color_continuous_scale=[
            "#f2d70b",
            "#a56eff",
            "#00CBC1"
        ],
        title=f"Wine Dataset - {ordering_name} axis order, colored by {color_title}"
    )
#hier legend einfügen

    fig.update_traces(
        dimensions=[
            dict(
                label=dim,
                values=df[dim],
                tickvals=[1, 2, 3],
                ticktext=["class 1", "class 2", "class 3"]
            ) if dim == "class" else dict(
                label=dim,
                values=df[dim]
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

#DASH / APP LAYOUT

app = Dash(__name__)

app.layout = html.Div(
    children=[
        html.H1("Improved Parallel Coordinates"),

        html.P(
            "Interactive Parallel Coordinates with KMeans clustering and axis reordering."
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
                {"label": "Original wine class", "value": "class"},
                {"label": "KMeans cluster", "value": "cluster"},
                ],
                value="cluster",
                clearable=False
                ),
        
        # explaining the drop down options
        html.Div(
            id="ordering-description",
            style={
                "marginTop": "15px",
                "marginBottom": "15px",
                "fontSize": "16px"
            }
        ),

        dcc.Graph(
            id="parallel-coordinates-plot",
            figure=create_parallel_coordinates("Original", "cluster")
        )
    ]
)


@app.callback(
    Output("parallel-coordinates-plot", "figure"),
    Output("ordering-description", "children"),
    Input("ordering-dropdown", "value"),
    Input("color-dropdown", "value")
)
def update_plot(selected_ordering, selected_color):
    descriptions = {
        "Original": "Original order: The axes are shown in the order of the dataset.",
        "SIM": "SIM order: Similar dimensions are placed next to each other.",
        "DIS": "DIS order: Dissimilar dimensions are placed next to each other. This can make clusters more salient in cluttered data."
    }

    return create_parallel_coordinates(selected_ordering, selected_color), descriptions[selected_ordering]

if __name__ == "__main__":
    app.run(debug=True)