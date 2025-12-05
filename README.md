# MNIST UMAP Visualization

UMAP visualization of activation embeddings in the MNISTDiffusion (Github: https://github.com/bot66/MNISTDiffusion) network.

## Versions

### Plotly Version (NEW)
Interactive Plotly/Dash implementation with enhanced features:
- Real-time image preview on hover
- Interactive UMAP parameter controls
- Responsive Bootstrap-based UI
- Class-based color coding

### Bokeh Version (Original)
Original implementation based on FairFace visualization ( https://github.com/SNaGLab/Fairface ) from Rahul Kumar ( https://github.com/rahulkumarm ).

## Setup

### Conda Environment
```bash
conda create -n plotly_viz python=3.12 -y
conda activate plotly_viz
pip install -r requirements.txt
```

## Run

### Plotly Version
```bash
./run_plotly_app.sh
# Or manually:
conda activate plotly_viz
python plotly_app.py
```
Access at http://localhost:8050


## Features

### Interactive Controls
- Method selection (UMAP)
- Number of neighbors (0-500)
- Minimum distance (0-1)
- Layer selection (midBlock2ave)

### Visualization
- Scatter plot with class-based coloring
- Hover for coordinates and class info
- Image preview panel
- Zoom, pan, and reset tools

## Data Structure
```
visualize/
├── data/
│   └── MNIST_train6k_midBlock2ave.csv
├── static/
│   └── images/
│       └── MNIST_train_images/
└── MNIST_diffuse6k_out.csv
```

## Development

### Before Committing
1. Run pylint: `pylint plotly_app.py`
2. Ensure tests pass
3. Create PR from feature branch
