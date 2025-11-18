"""
MNIST UMAP Visualization with Plotly/Dash
Converts Bokeh-based visualization to Plotly for interactive exploration
of MNIST diffusion network activations
"""

import base64
import json
import os
from io import BytesIO

import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, Patch, ALL, callback_context
from dash.exceptions import PreventUpdate
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
    # Hidden stores for state management
    dcc.Store(id='selected-point-store', data=None),
    dcc.Store(id='neighbor-indices-store', data=None),
    dcc.Store(id='activation-data-store', data=None),
    dcc.Store(id='layout-data-store', data=None),

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
            ], className="mb-3"),

            # Selection panel
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5("Selection", className="card-title d-inline"),
                        dbc.Button(
                            "✕",
                            id='clear-selection-button',
                            color="link",
                            size="sm",
                            className="float-end p-0",
                            style={'fontSize': '20px', 'lineHeight': '1', 'display': 'none'}
                        ),
                    ], className="mb-2"),
                    html.Div(id='selection-info', children=[
                        html.Div("Click a point to select", className="text-muted")
                    ], className="p-2 border rounded mb-3"),

                    html.Hr(className="my-3"),

                    html.H6("Neighbor Selection", className="card-subtitle mb-2"),

                    html.Label("Method"),
                    dcc.Dropdown(
                        id='neighbor-method-select',
                        options=[
                            {'label': 'K-Nearest Neighbors', 'value': 'KNN'}
                        ],
                        value='KNN',
                        className="mb-3",
                        disabled=True
                    ),

                    html.Div(id='knn-controls', children=[
                        html.Label("K (Number of Neighbors)"),
                        dcc.Slider(
                            id='k-neighbors-slider',
                            min=1,
                            max=100,
                            step=1,
                            value=25,
                            marks={i: str(i) for i in [1, 25, 50, 75, 100]},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-3",
                            disabled=True
                        ),
                    ]),

                    dbc.Button(
                        "Find Neighbors",
                        id='find-neighbors-button',
                        color="primary",
                        className="w-100 mb-3",
                        disabled=True
                    ),

                    html.H6("Neighbors Found", className="card-subtitle mb-2"),
                    html.Div(id='neighbor-list', children=[
                        html.Div("No neighbors found", className="text-muted")
                    ], className="p-2 border rounded",
                    style={'maxHeight': '300px', 'overflowY': 'auto'})
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
     Output('status-message', 'children'),
     Output('activation-data-store', 'data'),
     Output('layout-data-store', 'data')],
    [Input('update-button', 'n_clicks')],
    [State('method-select', 'value'),
     State('n-neighbors-slider', 'value'),
     State('min-dist-slider', 'value'),
     State('perplexity-slider', 'value'),
     State('learning-rate-slider', 'value'),
     State('layer-select', 'value'),
     State('activation-data-store', 'data'),
     State('layout-data-store', 'data')],
    prevent_initial_call=False
)
def update_plot(n_clicks,
                method, n_neighbors, min_dist, perplexity, learning_rate, layer,
                stored_activ, stored_layout):
    # pylint: disable=unused-argument,too-many-arguments,too-many-locals,too-many-statements
    """Update UMAP plot based on selected parameters."""

    status = f"Computing {method} with n_neighbors={n_neighbors}, min_dist={min_dist}..."

    # Load activation data
    activ = pd.read_csv(f'visualize/data/MNIST_train6k_{layer}.csv', header=None)
    print(f"Activation shape: {activ.shape}")

    # Drop index columns
    activ_clean = activ.drop(activ.columns[[0, 1]], axis=1)

    # Compute dimensionality reduction layout (always recalculate on update)
    if method == "UMAP":
        layout = umap.UMAP(
            metric="cosine",
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            low_memory=False,
            random_state=8675309,
            verbose=True
        ).fit_transform(activ_clean)
    elif method == "TSNE":
        layout = TSNE(
            n_components=2,
            metric="cosine",
            learning_rate=learning_rate,
            perplexity=min(perplexity, len(activ_clean) - 1),
            random_state=8675309,
            verbose=1
        ).fit_transform(activ_clean)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Normalize layout
    layout = normalize_layout(layout)
    layout_dict = {'x': layout[:, 0].tolist(), 'y': layout[:, 1].tolist()}

    # Store activation data for KNN computation
    activ_dict = activ_clean.values.tolist()

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

    return fig, status, activ_dict, layout_dict


@app.callback(
    Output('umap-plot', 'figure', allow_duplicate=True),
    [Input('selected-point-store', 'data'),
     Input('neighbor-indices-store', 'data'),
     Input('layout-data-store', 'data')],
    [State('umap-plot', 'figure')],
    prevent_initial_call=True
)
def update_plot_highlighting(selected_point, neighbor_indices, layout_data, current_figure):
    """Update plot to highlight selected point and neighbors."""
    # Don't update if no layout data or no figure exists yet
    if layout_data is None or current_figure is None:
        raise PreventUpdate

    n_points = len(layout_data['x'])

    # Create patch to update only marker properties
    patched_figure = Patch()

    # Prepare marker styling based on selection and neighbors
    marker_sizes = [5] * n_points
    marker_line_widths = [0] * n_points
    marker_line_colors = ['rgba(0,0,0,0)'] * n_points
    marker_opacities = [1.0] * n_points

    # Highlight selected point
    if selected_point is not None:
        idx = selected_point.get('idx')
        if idx is not None and 0 <= idx < n_points:
            marker_sizes[idx] = 15
            marker_line_widths[idx] = 3
            marker_line_colors[idx] = 'black'
            # Fade non-selected/non-neighbor points
            marker_opacities = [0.3] * n_points
            marker_opacities[idx] = 1.0

    # Highlight neighbors
    if neighbor_indices is not None and isinstance(neighbor_indices, list):
        for n_idx in neighbor_indices:
            if 0 <= n_idx < n_points:
                marker_sizes[n_idx] = 7
                marker_line_widths[n_idx] = 2
                marker_line_colors[n_idx] = 'black'
                marker_opacities[n_idx] = 1.0

    # Update the first trace (main scatter plot)
    patched_figure['data'][0]['marker']['size'] = marker_sizes
    patched_figure['data'][0]['marker']['opacity'] = marker_opacities
    patched_figure['data'][0]['marker']['line'] = {
        'width': marker_line_widths,
        'color': marker_line_colors
    }

    return patched_figure


@app.callback(
    [Output('selected-point-store', 'data'),
     Output('neighbor-method-select', 'disabled'),
     Output('k-neighbors-slider', 'disabled'),
     Output('find-neighbors-button', 'disabled'),
     Output('clear-selection-button', 'style'),
     Output('neighbor-indices-store', 'data', allow_duplicate=True)],
    [Input('umap-plot', 'clickData'),
     Input('clear-selection-button', 'n_clicks')],
    [State('selected-point-store', 'data'),
     State('neighbor-indices-store', 'data')],
    prevent_initial_call=True
)
def select_point(click_data, clear_clicks, selected_point, neighbor_indices):
    """Handle point selection on click and deselection on clear button."""
    # pylint: disable=unused-argument,too-many-return-statements,too-many-locals
    ctx = callback_context

    # Check which input triggered the callback
    if not ctx.triggered:
        clear_button_style = {'fontSize': '20px', 'lineHeight': '1', 'display': 'none'}
        return None, True, True, True, clear_button_style, None

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Clear button was clicked - clear selection and neighbors
    if trigger_id == 'clear-selection-button':
        clear_button_style = {'fontSize': '20px', 'lineHeight': '1', 'display': 'none'}
        return None, True, True, True, clear_button_style, None

    # Point was clicked
    if click_data is None:
        clear_button_style = {'fontSize': '20px', 'lineHeight': '1', 'display': 'none'}
        return None, True, True, True, clear_button_style, None

    try:
        point = click_data['points'][0]
        idx = int(point['customdata'][2])
        x = point['x']
        y = point['y']
        target = point['customdata'][1]

        # If no point is selected yet, select this one and clear neighbors
        if selected_point is None:
            selected_data = {
                'idx': idx,
                'x': float(x),
                'y': float(y),
                'target': int(float(target))
            }
            clear_button_style = {'fontSize': '20px', 'lineHeight': '1', 'display': 'block'}
            return selected_data, False, False, False, clear_button_style, None

        # If a point is already selected, toggle the clicked point as a neighbor
        # Initialize neighbor list if needed
        if neighbor_indices is None:
            neighbor_indices = []

        # Toggle neighbor: remove if present, add if not
        if idx in neighbor_indices:
            updated_neighbors = [n for n in neighbor_indices if n != idx]
        else:
            updated_neighbors = neighbor_indices + [idx]

        # Keep existing selection and update neighbors
        clear_button_style = {'fontSize': '20px', 'lineHeight': '1', 'display': 'block'}
        return selected_point, False, False, False, clear_button_style, updated_neighbors

    except (KeyError, IndexError, ValueError) as e:
        print(f"Error selecting point: {e}")
        clear_button_style = {'fontSize': '20px', 'lineHeight': '1', 'display': 'none'}
        return None, True, True, True, clear_button_style, None


@app.callback(
    Output('selection-info', 'children'),
    Input('selected-point-store', 'data')
)
def display_selection_info(selected_point):
    """Display selected point info."""
    if selected_point is None:
        return html.Div("Click a point to select", className="text-muted")

    try:
        idx = selected_point['idx']
        x = selected_point['x']
        y = selected_point['y']
        target = selected_point['target']

        image_path = f'visualize/static/images/MNIST_train_images/train{idx:0>5}.png'

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

    except (KeyError, FileNotFoundError, IOError) as e:
        return html.Div(f"Error: {str(e)}", className="text-danger")


@app.callback(
    Output('neighbor-indices-store', 'data'),
    [Input('find-neighbors-button', 'n_clicks')],
    [State('selected-point-store', 'data'),
     State('k-neighbors-slider', 'value'),
     State('layout-data-store', 'data')]
)
def find_neighbors(n_clicks, selected_point, k_value, layout_data):
    """Compute K-nearest neighbors for selected point."""
    if n_clicks is None or selected_point is None or layout_data is None:
        return None

    try:
        idx = selected_point['idx']

        # Convert stored layout data (2D UMAP/t-SNE coordinates) to numpy array
        layout_array = np.column_stack((layout_data['x'], layout_data['y']))

        # Compute KNN on 2D projection
        knn = NearestNeighbors(n_neighbors=k_value + 1, metric='euclidean')
        knn.fit(layout_array)

        # Find neighbors (k+1 because the point itself is included)
        _, indices = knn.kneighbors([layout_array[idx]])

        # Remove the point itself (first result)
        neighbor_indices = indices[0][1:].tolist()

        return neighbor_indices

    except (KeyError, IndexError, ValueError) as e:
        print(f"Error finding neighbors: {e}")
        return None


@app.callback(
    Output('neighbor-list', 'children'),
    [Input('neighbor-indices-store', 'data')],
    [State('layout-data-store', 'data')]
)
def display_neighbor_list(neighbor_indices, layout_data):
    """Display list of neighbors with images and info."""
    if neighbor_indices is None or not neighbor_indices:
        return html.Div("No neighbors found", className="text-muted")

    try:
        neighbor_items = []

        # Header showing count
        neighbor_items.append(
            html.Div([
                html.Strong(f"{len(neighbor_indices)} Neighbors"),
            ], className="mb-2")
        )

        # Create list of neighbor items
        for i, n_idx in enumerate(neighbor_indices):
            target = int(outputs["Target"].iloc[n_idx])
            image_path = f'visualize/static/images/MNIST_train_images/train{n_idx:0>5}.png'

            # Load and encode image
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    img.thumbnail((60, 60))
                    buffer = BytesIO()
                    img.save(buffer, format='PNG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    img_data = f'data:image/png;base64,{img_str}'
            else:
                img_data = None

            # Get coordinates from layout
            x_coord = layout_data['x'][n_idx] if layout_data else 0
            y_coord = layout_data['y'][n_idx] if layout_data else 0

            # Create neighbor card
            neighbor_items.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.Img(src=img_data, style={'width': '60px', 'height': '60px'})
                            if img_data else html.Div("No image", className="text-muted")
                        ], width=4),
                        dbc.Col([
                            html.P([
                                html.Strong(f"Neighbor {i+1}"), html.Br(),
                                f"Class: {target}", html.Br(),
                                f"X: {x_coord:.4f}", html.Br(),
                                f"Y: {y_coord:.4f}", html.Br(),
                                f"Index: {n_idx}"
                            ], className="mb-0 small")
                        ], width=7),
                        dbc.Col([
                            dbc.Button(
                                "\u2715",
                                id={'type': 'remove-neighbor-button', 'index': n_idx},
                                color="link",
                                size="sm",
                                className="p-0",
                                style={'fontSize': '20px', 'lineHeight': '1', 'color': '#dc3545'}
                            )
                        ], width=1, className="text-end")
                    ], className="align-items-center"),
                    html.Hr(className="my-2") if i < len(neighbor_indices) - 1 else None
                ], className="mb-2")
            )

        return html.Div(neighbor_items)

    except (KeyError, IndexError, FileNotFoundError, IOError) as e:
        return html.Div(f"Error: {str(e)}", className="text-danger")


@app.callback(
    Output('neighbor-indices-store', 'data', allow_duplicate=True),
    Input({'type': 'remove-neighbor-button', 'index': ALL}, 'n_clicks'),
    [State('neighbor-indices-store', 'data')],
    prevent_initial_call=True
)
def remove_neighbor(n_clicks, neighbor_indices):
    """Remove a specific neighbor from the list."""
    # pylint: disable=unused-argument
    ctx = callback_context

    if not ctx.triggered or neighbor_indices is None:
        raise PreventUpdate

    # Find which button was clicked
    triggered_id = ctx.triggered[0]['prop_id']
    if triggered_id == '.':
        raise PreventUpdate

    # Check if the triggered button actually has a value (was actually clicked)
    triggered_value = ctx.triggered[0].get('value')
    if triggered_value is None:
        raise PreventUpdate

    # Parse the button ID to get the index
    button_id_str = triggered_id.split('.')[0]
    button_id = json.loads(button_id_str)
    idx_to_remove = button_id['index']

    # Remove the neighbor from the list
    updated_neighbors = [idx for idx in neighbor_indices if idx != idx_to_remove]

    return updated_neighbors


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
