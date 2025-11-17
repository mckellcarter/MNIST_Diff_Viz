"""
MNIST UMAP Visualization with Plotly/Dash
Converts Bokeh-based visualization to Plotly for interactive exploration
of MNIST diffusion network activations
"""

import base64
import os
from io import BytesIO

import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from PIL import Image

# Set random seed for reproducibility
np.random.seed(8675309)


def normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1):
    """Removes outliers and scales layout to between [0,1]."""
    # Compute percentiles
    mins = np.percentile(layout, min_percentile, axis=0)
    maxs = np.percentile(layout, max_percentile, axis=0)

    # Add margins
    mins -= relative_margin * (maxs - mins)
    maxs += relative_margin * (maxs - mins)

    # Clip broadcasts
    clipped = np.clip(layout, mins, maxs)

    # Embed within [0,1] along both axes
    clipped -= clipped.min(axis=0)
    clipped /= clipped.max(axis=0)

    return clipped


def encode_image_to_base64(image_path):
    """Convert image to base64 string for embedding in hover template."""
    try:
        with Image.open(image_path) as img:
            # Resize for hover display
            img.thumbnail((120, 120))
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f'data:image/png;base64,{img_str}'
    except (FileNotFoundError, IOError) as e:
        print(f"Error loading image {image_path}: {e}")
        return None


# Initialize Dash app with Bootstrap theme
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load metadata
outputs = pd.read_csv('visualize/MNIST_diffuse6k_out.csv')
print(f"Outputs shape: {outputs.shape}")

# Create layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("MNIST UMAP Visualization", className="text-center mb-4"),
        ], width=12)
    ]),

    dbc.Row([
        # Controls column
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Controls", className="card-title"),

                    html.Label("Method"),
                    dcc.Dropdown(
                        id='method-select',
                        options=[
                            {'label': 'UMAP', 'value': 'UMAP'},
                            {'label': 't-SNE', 'value': 'TSNE'}
                        ],
                        value='UMAP',
                        className="mb-3"
                    ),

                    # UMAP-specific controls
                    html.Div(id='umap-controls', children=[
                        html.Label("Number of Neighbors"),
                        dcc.Slider(
                            id='n-neighbors-slider',
                            min=5,
                            max=500,
                            step=5,
                            value=50,
                            marks={i: str(i) for i in range(0, 501, 100)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-3"
                        ),

                        html.Label("Minimum Distance"),
                        dcc.Slider(
                            id='min-dist-slider',
                            min=0,
                            max=1,
                            step=0.1,
                            value=0.1,
                            marks={i/10: f'{i/10:.1f}' for i in range(0, 11, 2)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-3"
                        ),
                    ]),

                    # t-SNE-specific controls
                    html.Div(id='tsne-controls', children=[
                        html.Label("Perplexity"),
                        dcc.Slider(
                            id='perplexity-slider',
                            min=5,
                            max=100,
                            step=5,
                            value=50,
                            marks={i: str(i) for i in range(0, 101, 25)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-3"
                        ),

                        html.Label("Learning Rate"),
                        dcc.Slider(
                            id='learning-rate-slider',
                            min=10,
                            max=1000,
                            step=10,
                            value=200,
                            marks={i: str(i) for i in [10, 200, 500, 1000]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-3"
                        ),
                    ], style={'display': 'none'}),

                    html.Label("Layer"),
                    dcc.Dropdown(
                        id='layer-select',
                        options=[{'label': 'midBlock2ave', 'value': 'midBlock2ave'}],
                        value='midBlock2ave',
                        className="mb-3"
                    ),

                    dbc.Button(
                        "Update Visualization",
                        id='update-button',
                        color="primary",
                        className="w-100 mt-3"
                    ),

                    html.Div(id='status-message', className="mt-3 text-muted"),

                    html.Hr(className="my-3"),

                    html.H6("Hover Info", className="card-subtitle mb-2"),
                    html.Div(id='hover-info', className="p-2 border rounded")
                ])
            ], className="h-100")
        ], width=3),

        # Visualization column
        dbc.Col([
            dcc.Loading(
                id="loading",
                type="default",
                children=[
                    dcc.Graph(
                        id='umap-plot',
                        style={'height': '800px'},
                        config={'displayModeBar': True, 'scrollZoom': True}
                    )
                ]
            )
        ], width=9)
    ])
], fluid=True, className="p-4")


@app.callback(
    [Output('umap-controls', 'style'),
     Output('tsne-controls', 'style')],
    [Input('method-select', 'value')]
)
def toggle_controls(method):
    """Show/hide controls based on selected method."""
    if method == 'UMAP':
        return {'display': 'block'}, {'display': 'none'}
    if method == 'TSNE':
        return {'display': 'none'}, {'display': 'block'}
    return {'display': 'block'}, {'display': 'none'}


@app.callback(
    [Output('umap-plot', 'figure'),
     Output('status-message', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('method-select', 'value'),
     State('n-neighbors-slider', 'value'),
     State('min-dist-slider', 'value'),
     State('perplexity-slider', 'value'),
     State('learning-rate-slider', 'value'),
     State('layer-select', 'value')],
    prevent_initial_call=False
)
def update_plot(n_clicks, method, n_neighbors, min_dist, perplexity, learning_rate, layer):
    # pylint: disable=unused-argument
    """Update UMAP plot based on selected parameters."""

    status = f"Computing {method} with n_neighbors={n_neighbors}, min_dist={min_dist}..."

    # Load activation data
    activ = pd.read_csv(f'visualize/data/MNIST_train6k_{layer}.csv', header=None)
    print(f"Activation shape: {activ.shape}")

    # Drop index columns
    activ.drop(activ.columns[[0, 1]], axis=1, inplace=True)

    # Compute dimensionality reduction layout
    if method == "UMAP":
        layout = umap.UMAP(
            metric="cosine",
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            low_memory=False,
            random_state=8675309,
            verbose=True
        ).fit_transform(activ)
    elif method == "TSNE":
        layout = TSNE(
            n_components=2,
            metric="cosine",
            learning_rate=learning_rate,
            perplexity=min(perplexity, len(activ) - 1),
            random_state=8675309,
            verbose=1
        ).fit_transform(activ)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize layout
    layout = normalize_layout(layout)

    # Prepare data for plotting
    x_coords = layout[:, 0]
    y_coords = layout[:, 1]
    targets = outputs["Target"].values

    # Create image paths
    image_paths = [f'visualize/static/images/MNIST_train_images/train{i:0>5}.png'
                   for i in range(layout.shape[0])]

    # Create custom hover data
    hover_text = []
    for i, (x, y, target) in enumerate(zip(x_coords, y_coords, targets)):
        hover_text.append(
            f"<b>Class:</b> {target}<br>"
            f"<b>X:</b> {x:.4f}<br>"
            f"<b>Y:</b> {y:.4f}<br>"
            f"<b>File:</b> train{i:0>5}.png<br>"
            f"<extra></extra>"
        )

    # Create scatter plot
    fig = go.Figure()

    # Use discrete colormap for 10 categories
    colors = px.colors.qualitative.G10
    target_colors = [colors[int(t) % len(colors)] for t in targets]

    # Add scatter points colored by target class
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers',
        marker={'size': 5, 'color': target_colors, 'line': {'width': 0}},
        text=hover_text,
        hovertemplate='%{text}',
        customdata=np.column_stack((image_paths, targets,
                                   np.arange(layout.shape[0]))),
    ))

    # Add discrete legend for classes
    for i in range(10):
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker={'size': 8, 'color': colors[i]},
            showlegend=True,
            name=f'Class {i}'
        ))

    # Update layout
    fig.update_layout(
        title=f"{method} Projection of MNIST Activation Embeddings",
        xaxis_title="UMAP Dimension 1",
        yaxis_title="UMAP Dimension 2",
        hovermode='closest',
        plot_bgcolor='white',
        height=800,
        xaxis={'gridcolor': 'lightgray'},
        yaxis={'gridcolor': 'lightgray'}
    )

    status = f"Visualization complete: {layout.shape[0]} points plotted"

    return fig, status


@app.callback(
    Output('hover-info', 'children'),
    Input('umap-plot', 'hoverData')
)
def display_hover_image(hover_data):
    """Display image and info for hovered point."""
    if hover_data is None:
        return html.Div("Hover over a point to see image", className="text-muted")

    try:
        point = hover_data['points'][0]
        image_path = point['customdata'][0]
        target = point['customdata'][1]
        idx = int(point['customdata'][2])
        x = point['x']
        y = point['y']

        # Load and encode image
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                img.thumbnail((120, 120))
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                img_data = f'data:image/png;base64,{img_str}'
        else:
            img_data = None

        # Create info display
        return html.Div([
            html.Img(src=img_data, style={'width': '120px', 'height': '120px'})
            if img_data else html.Div("Image not found"),
            html.Hr(className="my-2"),
            html.P([
                html.Strong("Class: "), f"{target}", html.Br(),
                html.Strong("X: "), f"{x:.4f}", html.Br(),
                html.Strong("Y: "), f"{y:.4f}", html.Br(),
                html.Strong("File: "), f"train{idx:0>5}.png"
            ], className="mb-0 small")
        ])

    except (KeyError, IndexError, FileNotFoundError, IOError) as e:
        return html.Div(f"Error: {str(e)}", className="text-danger")


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
