# MNIST Diffusion Visualization - Project Documentation

## Project Overview
Interactive UMAP/t-SNE visualization of MNIST diffusion network activation embeddings using Plotly/Dash. Migrated from original Bokeh implementation.

**Source**: MNISTDiffusion (https://github.com/bot66/MNISTDiffusion)
**Original**: FairFace viz (https://github.com/SNaGLab/Fairface)

## Tech Stack
- **Python 3.12**
- **Dash/Plotly**: Interactive web visualization
- **UMAP/t-SNE**: Dimensionality reduction
- **Pandas/NumPy**: Data processing
- **Pillow**: Image handling
- **Bootstrap**: UI styling

## Project Structure
```
MNIST_Diff_Viz/
├── plotly_app.py              # Main Dash application
├── run_plotly_app.sh          # Launch script
├── requirements.txt           # Python dependencies
├── README.md                  # User documentation
├── PROJECT.md                 # This file
└── visualize/
    ├── data/
    │   └── MNIST_train6k_midBlock2ave.csv  # Activation data (6000 samples)
    ├── static/images/MNIST_train_images/   # PNG thumbnails (train00000-05999.png)
    └── MNIST_diffuse6k_out.csv             # Metadata (Target labels)
```

## Key Components

### 1. Data Pipeline
- **Activation data**: `visualize/data/MNIST_train6k_midBlock2ave.csv` (6000×N features)
- **Metadata**: `visualize/MNIST_diffuse6k_out.csv` (Target class labels 0-9)
- **Images**: `visualize/static/images/MNIST_train_images/train{i:0>5}.png`

### 2. Core Functions

#### `normalize_layout(layout, min_percentile=1, max_percentile=99, relative_margin=0.1)`
- Removes outliers via percentile clipping
- Scales to [0,1] range
- Adds relative margins

#### `encode_image_to_base64(image_path)`
- Converts PNG to base64 for hover display
- Resizes to 120×120 thumbnail
- Returns data URI string

### 3. Dimensionality Reduction Methods

#### UMAP (default)
- **Metric**: cosine similarity
- **Parameters**:
  - n_neighbors: 5-500 (default: 50)
  - min_dist: 0-1 (default: 0.1)
- **Random state**: 8675309 (reproducibility)

#### t-SNE
- **Metric**: cosine similarity
- **Parameters**:
  - perplexity: 5-100 (default: 50)
  - learning_rate: 10-1000 (default: 200)
- **Random state**: 8675309

### 4. Dash Callbacks

#### `toggle_controls(method)`
**Triggers**: `method-select` dropdown
**Updates**: Show/hide UMAP vs t-SNE controls

#### `update_plot(n_clicks, method, n_neighbors, min_dist, perplexity, learning_rate, layer)`
**Triggers**: `update-button` click
**Updates**: Main scatter plot + status message
**Process**:
1. Load activation CSV
2. Drop index columns (0, 1)
3. Compute UMAP/t-SNE embedding
4. Normalize layout
5. Build Plotly scatter with class colors
6. Add discrete legend (10 classes)

#### `display_hover_image(hover_data)`
**Triggers**: Hover over scatter point
**Updates**: `hover-info` panel
**Displays**: 120×120 thumbnail + class/coords/filename

### 5. UI Layout
- **Left sidebar (3 cols)**: Controls + hover info panel
- **Right panel (9 cols)**: 800px scatter plot
- **Color scheme**: 10-class discrete palette (Plotly G10)
- **Hover**: Custom template with embedded image data

## Environment Setup

### Conda
```bash
conda create -n plotly_viz python=3.12 -y
conda activate plotly_viz
pip install -r requirements.txt
```

### Run
```bash
./run_plotly_app.sh
# Access: http://localhost:8050
```

## Development Workflow

### Before Committing
1. **Lint**: `pylint plotly_app.py` (must pass)
2. **Test**: Use Puppeteer MCP tools for UI testing
3. **Branch**: `feature/description` (never commit to main)
4. **PR**: Required for all changes

### Git Branches
- `main`: Stable production
- `feature/plotly-conversion`: Bokeh→Plotly migration (active)

## Current Features
✅ Interactive UMAP/t-SNE controls
✅ Real-time hover image preview
✅ Class-based color coding (0-9)
✅ Responsive Bootstrap UI
✅ Zoom/pan/reset tools
✅ Status messages for computation

## Potential Features for Future Sessions

### High Priority
- [ ] **Layer selection dropdown**: Add support for multiple activation layers
- [ ] **Data upload**: Allow custom CSV/image dataset uploads
- [ ] **Export**: Download plot as PNG/SVG or layout CSV
- [ ] **Permalink**: URL params for sharing specific configurations
- [ ] **Annotation tools**: Click to label/group points

### Medium Priority
- [ ] **Comparison mode**: Side-by-side UMAP vs t-SNE
- [ ] **Animation**: Transition between parameter settings
- [ ] **Filtering**: Show/hide specific classes
- [ ] **Search**: Find specific image by index/class
- [ ] **3D mode**: Toggle 3D UMAP/t-SNE projection

### Low Priority
- [ ] **Batch processing**: Process full 60k MNIST dataset
- [ ] **PCA baseline**: Add PCA for comparison
- [ ] **Metrics dashboard**: Silhouette score, Trustworthiness
- [ ] **Clustering overlay**: K-means/DBSCAN boundaries
- [ ] **Dark mode**: Toggle UI theme

## Code Quality Standards
- **Pylint**: Must pass before commits
- **Docstrings**: All functions documented
- **Type hints**: Encouraged for new code
- **Random seed**: Fixed at 8675309 for reproducibility
- **Error handling**: Try/except for file I/O

## Dependencies Matrix
| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Array operations |
| pandas | 2.2.0 | CSV handling |
| umap-learn | 0.5.5 | UMAP algorithm |
| scikit-learn | 1.4.0 | t-SNE, utilities |
| plotly | 5.18.0 | Interactive plots |
| dash | 2.14.2 | Web framework |
| dash-bootstrap-components | 1.5.0 | UI components |
| pillow | 10.2.0 | Image processing |
| pylint | 3.0.3 | Linting |

## Known Limitations
- Single layer support (midBlock2ave only)
- 6k sample subset (full dataset is 60k)
- No persistent state (recalculates on every update)
- Image loading errors not gracefully handled in batch

## Performance Notes
- **UMAP**: ~2-5s for 6k samples (n_neighbors=50)
- **t-SNE**: ~10-20s for 6k samples (perplexity=50)
- **Image encoding**: On-demand (not preloaded)
- **Memory**: ~1GB for activation data + embeddings

## Testing Strategy
Use Puppeteer MCP tools to verify:
1. Initial page load (http://localhost:8050)
2. Method dropdown toggle (UMAP ↔ t-SNE controls)
3. Slider interactions (n_neighbors, min_dist, etc.)
4. Update button click (plot refresh)
5. Hover interactions (image preview appears)
6. Responsive layout (resize browser)

## Contact & Resources
- **GitHub Issues**: Track bugs/features
- **Docs**: README.md for user-facing instructions
- **Pylint config**: Default settings (3.0.3)
