"""
Configuration and Constants for Block Model Visualizer

This module contains all configuration values and constants used throughout the application.
"""

from typing import List, Dict

# ==============================================================================
# APPLICATION METADATA
# ==============================================================================
APP_VERSION = "1.0"
APP_TITLE = "Block Model Visualizer"
APP_ICON = "📦"

# ==============================================================================
# DEFAULT VALUES
# ==============================================================================
DEFAULT_SKIP_ROWS = 3
DEFAULT_OPACITY = 1.0
DEFAULT_MARKER_SIZE = 4
DEFAULT_MAX_BLOCKS = None  # None = render all blocks

# ==============================================================================
# COLORSCALES
# ==============================================================================
AVAILABLE_COLORSCALES: List[str] = [
    "Viridis",
    "Plasma",
    "Inferno",
    "Hot",
    "Portland",
    "RdYlBu",
    "Spectral",
    "Jet"
]

DEFAULT_COLORSCALE = "Viridis"

# ==============================================================================
# COLOR MODE MAPPING
# ==============================================================================
COLOR_MODE_OPTIONS: List[str] = [
    "Auto-detect",
    "Gradient (Numeric)",
    "Discrete (Categorical)"
]

COLOR_MODE_MAPPING: Dict[str, str] = {
    "Auto-detect": "auto",
    "Gradient (Numeric)": "gradient",
    "Discrete (Categorical)": "discrete"
}

# ==============================================================================
# CALCULATION MODES
# ==============================================================================
CALC_MODE_OPTIONS: List[str] = [
    "Calculate Block Sum",
    "Calculate Block Average",
    "Calculate Thickness",
    "Calculate Stripping Ratio"
]

CALC_MODE_MAPPING: Dict[str, str] = {
    "Calculate Thickness": "thickness",
    "Calculate Stripping Ratio": "stripping_ratio",
    "Calculate Block Sum": "block_sum",
    "Calculate Block Average": "block_average"
}

# ==============================================================================
# ATTRIBUTE FILTERING
# ==============================================================================
# Keywords to exclude from numeric attribute selection
EXCLUDE_ATTRIBUTE_KEYWORDS: List[str] = [
    'ijk', 'index', 'morig', 'nx', 'ny', 'nz',
    'centroid_x', 'centroid_y', 'centroid_z',
    'xc', 'yc', 'zc', 'dim_x', 'dim_y', 'dim_z'
]

# ==============================================================================
# FILE SETTINGS
# ==============================================================================
SUPPORTED_FILE_TYPES: List[str] = ['csv']
TEMP_UPLOAD_FILE = "temp_uploaded.csv"

# ==============================================================================
# THEME COLORS
# ==============================================================================
PRIMARY_GRADIENT_START = "#667eea"
PRIMARY_GRADIENT_END = "#764ba2"

# ==============================================================================
# VISUALIZATION SETTINGS
# ==============================================================================
DEFAULT_PLOT_WIDTH = 1200
DEFAULT_PLOT_HEIGHT = 800
DEFAULT_CAMERA_EYE = dict(x=1.5, y=1.5, z=1.5)

# ==============================================================================
# BLOCK MODEL DEFAULT DIMENSIONS
# ==============================================================================
DEFAULT_BLOCK_DX = 12.5
DEFAULT_BLOCK_DY = 12.5
DEFAULT_BLOCK_DZ = 1.0
