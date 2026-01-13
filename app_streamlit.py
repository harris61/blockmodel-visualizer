"""
Streamlit Web Interface for Block Model Column Calculator

Interactive web app to:
- Select/upload CSV files
- Choose visualization type (point cloud or boxes)
- Select attributes interactively
- Adjust parameters with sliders
- Real-time 3D visualization

Usage:
    streamlit run app_streamlit.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from pathlib import Path
import os
from PIL import Image
import base64
from io import BytesIO
import hashlib
import json

from block_model_visualizer import BlockModelVisualizer
from styles import get_custom_css
from templates import (
    get_header_html,
    get_3d_controls_html,
    get_important_notes_html,
    get_user_guide_header_html,
    get_step1_html,
    get_step2_html,
    get_step3_html,
    get_step4_html,
    get_footer_html
)
from config import (
    APP_TITLE,
    APP_ICON,
    DEFAULT_SKIP_ROWS,
    DEFAULT_OPACITY,
    DEFAULT_MARKER_SIZE,
    AVAILABLE_COLORSCALES,
    COLOR_MODE_OPTIONS,
    COLOR_MODE_MAPPING,
    EXCLUDE_ATTRIBUTE_KEYWORDS,
    SUPPORTED_FILE_TYPES,
    TEMP_UPLOAD_FILE
)
from ui_components import (
    initialize_session_state,
    render_block_sum_config,
    render_block_sum_controls,
    apply_vertical_sum
)


# Page config
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Display header
st.markdown(get_header_html(), unsafe_allow_html=True)

# Sidebar header (app info + credits)
img = Image.open("2.png")
buffered = BytesIO()
img.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

st.sidebar.markdown(get_footer_html(img_str), unsafe_allow_html=True)

# Sidebar for file selection and parameters

# Use point cloud visualization by default (3D boxes removed due to performance)
viz_type = "Point Cloud (Fast)"

# ============================================================================
# FILE SELECTION
# ============================================================================
st.sidebar.markdown("#### File Upload")

csv_file = None
uploaded_file = st.sidebar.file_uploader(
    "Upload your Block Model .csv file",
    type=SUPPORTED_FILE_TYPES,
    help="Upload block model CSV file (Datamine, Vulcan, or standard format)"
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = Path(TEMP_UPLOAD_FILE)
    # Remove old temp file if exists
    if temp_path.exists():
        try:
            temp_path.unlink()
        except Exception:
            pass  # Ignore errors if file is in use

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    csv_file = str(temp_path)
    st.sidebar.success(f"Loaded: {uploaded_file.name}")

# Skip rows parameter
skip_rows = st.sidebar.number_input(
    "Number of Header Lines",
    min_value=0,
    max_value=10,
    value=DEFAULT_SKIP_ROWS,
    help="Number of Header Lines"
)

# ============================================================================
# PARAMETERS
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("#### Parameters")

# Process all blocks by default (no sampling)
sample_size = None
max_blocks = None

# Default values (no UI controls)
opacity = DEFAULT_OPACITY
marker_size = DEFAULT_MARKER_SIZE

colorscale = st.sidebar.selectbox(
    "Colorscale:",
    AVAILABLE_COLORSCALES,
    help="Color scheme for visualization (gradient mode)"
)

# Color mode (gradient vs discrete)
st.sidebar.markdown("")
color_mode = st.sidebar.radio(
    "Color scheme type:",
    COLOR_MODE_OPTIONS,
    index=0,
    help="Auto-detect: Automatically choose based on data type\n"
         "Gradient: Continuous colorscale for numeric ranges\n"
         "Discrete: Distinct colors for categories"
)

# Map to API parameter
color_mode_param = COLOR_MODE_MAPPING[color_mode]

# ============================================================================
# CACHED DATA LOADING FUNCTION
# ============================================================================
@st.cache_data(show_spinner=False)
def load_blockmodel_data(csv_path: str, skip_rows: int, cache_version: int = 2):
    """
    Cached data loading function to prevent reloading data on every parameter change.
    Cache is invalidated only when csv_path or skip_rows changes.

    Args:
        csv_path: Path to CSV file
        skip_rows: Number of header rows to skip
        cache_version: Version number to invalidate cache when data structure changes
    """
    viz = BlockModelVisualizer(csv_path, skip_rows=skip_rows)
    viz.load_data()
    return viz

@st.cache_data(show_spinner=False)
def compute_block_sum(df_dict: dict, coord_cols: dict, dim_cols: dict, config_hash: str, config: dict,
                       metadata_lines: list, original_header: list, original_column_order: list,
                       csv_file: str, skip_rows: int):
    """
    Cached block sum computation to prevent recalculation on every parameter change.
    Cache is invalidated only when the configuration changes.

    Args:
        df_dict: Dictionary representation of the dataframe
        coord_cols: Coordinate column mapping
        dim_cols: Dimension column mapping
        config_hash: Hash of configuration to use as cache key
        config: Configuration dictionary
        metadata_lines: Original metadata lines from CSV
        original_header: Original header column names
        original_column_order: Original column order
        csv_file: Path to CSV file
        skip_rows: Number of header rows to skip

    Returns:
        dict: Dictionary with processed dataframe and all attributes
    """
    # Reconstruct dataframe and visualizer
    temp_df = pd.DataFrame(df_dict)
    temp_viz = BlockModelVisualizer.__new__(BlockModelVisualizer)
    temp_viz.df = temp_df
    temp_viz.coord_cols = coord_cols
    temp_viz.dim_cols = dim_cols

    # Restore header-related attributes for proper export
    temp_viz.metadata_lines = metadata_lines
    temp_viz.original_header = original_header
    temp_viz.original_column_order = original_column_order
    temp_viz.csv_file = Path(csv_file)
    temp_viz.skip_rows = skip_rows

    # Apply vertical sum
    apply_vertical_sum(temp_viz, config)

    # Return all necessary data to preserve state
    return {
        'df': temp_viz.df,
        'coord_cols': temp_viz.coord_cols,
        'dim_cols': temp_viz.dim_cols,
        'metadata_lines': temp_viz.metadata_lines,
        'original_header': temp_viz.original_header,
        'original_column_order': temp_viz.original_column_order
    }

# ============================================================================
# MAIN AREA - DATA LOADING AND VISUALIZATION
# ============================================================================

if csv_file is not None:
    try:
        # ====================================================================
        # STAGE 1: Load data (cached)
        # ====================================================================
        with st.spinner("Stage 1/3: Loading data from CSV..."):
            viz = load_blockmodel_data(csv_file, skip_rows, cache_version=2)

        # Data loaded successfully (cache handled by @st.cache_data decorator)
        st.sidebar.success("Data loaded successfully")

        # ====================================================================
        # BLOCK SUM FEATURE
        # ====================================================================
        # Initialize session state
        initialize_session_state(viz)

        # Render block sum configuration UI
        config = render_block_sum_config(viz)

        # ====================================================================
        # STAGE 2: Apply block sum if enabled (cached)
        # ====================================================================
        if st.session_state.is_summed:
            # Create a hash of the config to use as cache key
            config_str = json.dumps(st.session_state.block_sum_config, sort_keys=True, default=str)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()

            # Convert original df to dict for caching
            df_dict = st.session_state.original_df.to_dict('list')

            # Use cached computation (cache handled by @st.cache_data decorator)
            with st.spinner("Stage 2/3: Processing block calculations..."):
                result = compute_block_sum(
                    df_dict,
                    viz.coord_cols,
                    viz.dim_cols,
                    config_hash,
                    st.session_state.block_sum_config,
                    viz.metadata_lines,
                    viz.original_header,
                    viz.original_column_order,
                    str(viz.csv_file),
                    viz.skip_rows
                )

                # Update viz with ALL returned attributes to preserve state
                viz.df = result['df']
                viz.coord_cols = result['coord_cols']
                viz.dim_cols = result['dim_cols']
                viz.metadata_lines = result['metadata_lines']
                viz.original_header = result['original_header']
                viz.original_column_order = result['original_column_order']

            # Calculation completed
            st.sidebar.info("Block calculations completed")

        # Render control buttons and handle actions
        action = render_block_sum_controls(viz)
        if action in ['calculate', 'reset']:
            st.rerun()  # Rerun to update visualization

        # ====================================================================
        # ATTRIBUTE SELECTION
        # ====================================================================
        st.markdown("## Attribute Selection")
        st.markdown("---")

        # Get numeric columns
        numeric_cols = viz.df.select_dtypes(include=[np.number]).columns.tolist()
        available_attrs = [col for col in numeric_cols
                          if not any(kw in col.lower() for kw in EXCLUDE_ATTRIBUTE_KEYWORDS)]

        # Get categorical columns
        categorical_cols = viz.df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

        # Radio button to choose attribute type
        attr_type = st.radio(
            "Pilih tipe attribute:",
            ["Numeric", "Categorical"],
            horizontal=True,
            key="attr_type_radio"
        )

        # Attribute selection with dropdowns
        col1, col2 = st.columns(2)

        with col1:
            if available_attrs:
                numeric_attr = st.selectbox(
                    "Numeric Attributes:",
                    available_attrs,
                    index=0,
                    key="numeric_select",
                    disabled=(attr_type != "Numeric")
                )
            else:
                numeric_attr = None

        with col2:
            if categorical_cols:
                categorical_attr = st.selectbox(
                    "Categorical Attributes:",
                    categorical_cols,
                    index=0,
                    key="categorical_select",
                    disabled=(attr_type != "Categorical")
                )
            else:
                categorical_attr = None

        # Determine which attribute to use based on radio selection
        if attr_type == "Numeric":
            color_attr = numeric_attr
        else:  # Categorical
            color_attr = categorical_attr

        selected_attrs = [color_attr] if color_attr else []

        # ====================================================================
        # FILTERING (DEBOUNCED with Apply Button)
        # ====================================================================
        with st.expander("Data Filtering (Optional)"):
            st.write("Filter blocks by attribute values:")

            filter_enabled = st.checkbox("Enable filtering")

            if filter_enabled:
                filter_attr = st.selectbox(
                    "Filter by attribute:",
                    available_attrs,
                    key="filter_attr"
                )

                filter_type = st.radio(
                    "Filter type:",
                    ["Range", "Minimum value", "Maximum value"],
                    horizontal=True
                )

                # Get data for sliders (use original_df if available to get full range)
                source_df = st.session_state.get('original_df', viz.df)
                attr_min = float(source_df[filter_attr].min())
                attr_max = float(source_df[filter_attr].max())

                # Initialize session state for filter values
                if 'filter_dict' not in st.session_state:
                    st.session_state.filter_dict = None
                if 'filter_applied' not in st.session_state:
                    st.session_state.filter_applied = False

                # Store slider values in temporary variables (doesn't trigger filter yet)
                if filter_type == "Range":
                    filter_range = st.slider(
                        f"{filter_attr} range:",
                        min_value=attr_min,
                        max_value=attr_max,
                        value=(attr_min, attr_max),
                        key="filter_range_slider"
                    )
                    temp_filter_dict = {filter_attr: filter_range}
                elif filter_type == "Minimum value":
                    filter_min = st.slider(
                        f"Minimum {filter_attr}:",
                        min_value=attr_min,
                        max_value=attr_max,
                        value=attr_min,
                        key="filter_min_slider"
                    )
                    temp_filter_dict = {filter_attr: (filter_min, None)}
                else:
                    filter_max = st.slider(
                        f"Maximum {filter_attr}:",
                        min_value=attr_min,
                        max_value=attr_max,
                        value=attr_max,
                        key="filter_max_slider"
                    )
                    temp_filter_dict = {filter_attr: (None, filter_max)}

                # Add Apply and Clear buttons
                col_filter1, col_filter2 = st.columns(2)

                with col_filter1:
                    if st.button("Apply Filter", use_container_width=True):
                        st.session_state.filter_dict = temp_filter_dict
                        st.session_state.filter_applied = True

                with col_filter2:
                    if st.button("Clear Filter", use_container_width=True):
                        st.session_state.filter_dict = None
                        st.session_state.filter_applied = False

                # Apply filter if button was clicked
                if st.session_state.filter_applied and st.session_state.filter_dict:
                    viz.filter_data(st.session_state.filter_dict)
                    st.info(f"Filtered to {len(viz.df):,} blocks")
            else:
                # Clear filter state when filtering is disabled
                st.session_state.filter_dict = None
                st.session_state.filter_applied = False

        # ====================================================================
        # AUTO-GENERATE VISUALIZATION
        # ====================================================================
        st.markdown("## 3D Point Cloud Visualization")
        st.markdown("---")

        if len(selected_attrs) == 0 or color_attr is None:
            st.info("Please select an attribute from the dropdown above to visualize")
        else:
            # ================================================================
            # STAGE 3: Generate visualization
            # ================================================================
            with st.spinner(f"Stage 3/3: Creating 3D visualization for {len(viz.df):,} blocks..."):
                # Create point cloud visualization
                fig = viz.visualize_scatter(
                    color_by=color_attr,
                    marker_size=marker_size,
                    opacity=opacity,
                    colorscale=colorscale,
                    title=f"Block Model - {color_attr} ({len(viz.df):,} blocks)"
                )

            # Show completion status
            st.sidebar.success(f"Visualization ready ({len(viz.df):,} blocks)")

            # Display
            st.plotly_chart(fig, use_container_width=True)

            # 3D Controls info
            st.markdown(get_3d_controls_html(), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())

else:
    # Important Notes
    st.markdown(get_important_notes_html(), unsafe_allow_html=True)

    # User guide header
    st.markdown(get_user_guide_header_html(), unsafe_allow_html=True)

    # User guide steps
    st.markdown(get_step1_html(), unsafe_allow_html=True)
    st.markdown(get_step2_html(), unsafe_allow_html=True)
    st.markdown(get_step3_html(), unsafe_allow_html=True)
    st.markdown(get_step4_html(), unsafe_allow_html=True)

# Footer removed from sidebar; header shows app info now.
