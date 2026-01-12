"""
Streamlit Web Interface for Block Model Visualizer

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
    CALC_MODE_OPTIONS,
    CALC_MODE_MAPPING,
    EXCLUDE_ATTRIBUTE_KEYWORDS,
    SUPPORTED_FILE_TYPES,
    TEMP_UPLOAD_FILE
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

# Sidebar for file selection and parameters

# Use point cloud visualization by default (3D boxes removed due to performance)
viz_type = "Point Cloud (Fast)"

# ============================================================================
# FILE SELECTION
# ============================================================================
st.sidebar.markdown("#### 📁 File Upload")

csv_file = None
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=SUPPORTED_FILE_TYPES,
    help="Upload block model CSV file (Datamine, Vulcan, or standard format)"
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = Path(TEMP_UPLOAD_FILE)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    csv_file = str(temp_path)
    st.sidebar.success(f"✓ Loaded: {uploaded_file.name}")

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
st.sidebar.markdown("#### ⚙️ Parameters")

# Process all blocks by default (no sampling)
sample_size = None
max_blocks = None

# Default values (no UI controls)
opacity = DEFAULT_OPACITY
marker_size = DEFAULT_MARKER_SIZE

colorscale = st.sidebar.selectbox(
    "🎨 Colorscale:",
    AVAILABLE_COLORSCALES,
    help="Color scheme for visualization (gradient mode)"
)

# Color mode (gradient vs discrete)
st.sidebar.markdown("")
color_mode = st.sidebar.radio(
    "🖌️ Color scheme type:",
    COLOR_MODE_OPTIONS,
    index=0,
    help="Auto-detect: Automatically choose based on data type\n"
         "Gradient: Continuous colorscale for numeric ranges\n"
         "Discrete: Distinct colors for categories"
)

# Map to API parameter
color_mode_param = COLOR_MODE_MAPPING[color_mode]

# ============================================================================
# MAIN AREA - DATA LOADING AND VISUALIZATION
# ============================================================================

if csv_file is not None:
    try:
        # Load data
        with st.spinner("Loading data..."):
            viz = BlockModelVisualizer(csv_file, skip_rows=skip_rows)
            viz.load_data()

        # ====================================================================
        # BLOCK SUM FEATURE
        # ====================================================================
        # Initialize session state for original data
        if 'original_df' not in st.session_state:
            st.session_state.original_df = viz.df.copy()
            st.session_state.is_summed = False

        # Block Sum Configuration
        st.markdown("## ⚙️ Block Sum Configuration")
        st.markdown("---")

        # Calculation Mode
        calc_mode = st.radio(
            "📊 Calculation Mode:",
            CALC_MODE_OPTIONS,
            horizontal=False,
            help="Thickness: Calculate thickness by category | SR: OB/Ore ratio | Block Sum: Sum attribute values by category | Block Average: Average attribute values by category"
        )

        # Get categorical columns
        cat_cols = viz.df.select_dtypes(include=['object']).columns.tolist()
        # Get numeric columns for Block Sum/Average
        numeric_cols = viz.df.select_dtypes(include=[np.number]).columns.tolist()

        categorical_attr = None
        selected_categories = None
        ob_categories = None
        ore_categories = None
        value_attr = None

        if calc_mode in ["Calculate Thickness", "Calculate Stripping Ratio", "Calculate Block Sum", "Calculate Block Average"] and cat_cols:
            # Select categorical attribute
            categorical_attr = st.selectbox(
                "Select Categorical Attribute:",
                cat_cols,
                help="Choose attribute for classification (e.g., lithology, zone, rock_type)"
            )

            if categorical_attr:
                # Get unique categories
                unique_cats = viz.df[categorical_attr].unique().tolist()

                if calc_mode == "Calculate Thickness":
                    selected_categories = st.multiselect(
                        "Select Categories to Calculate Thickness:",
                        unique_cats,
                        default=unique_cats[:2] if len(unique_cats) >= 2 else unique_cats,
                        help="Select one or more categories to calculate thickness"
                    )

                elif calc_mode == "Calculate Stripping Ratio":
                    col1, col2 = st.columns(2)

                    with col1:
                        ob_categories = st.multiselect(
                            "OB/Waste Categories:",
                            unique_cats,
                            help="Select categories considered as Overburden/Waste"
                        )

                    with col2:
                        ore_categories = st.multiselect(
                            "Ore Categories:",
                            unique_cats,
                            help="Select categories considered as Ore"
                        )

                elif calc_mode in ["Calculate Block Sum", "Calculate Block Average"]:
                    # Select categories to filter
                    selected_categories = st.multiselect(
                        "Select Categories to Calculate:",
                        unique_cats,
                        default=unique_cats[:2] if len(unique_cats) >= 2 else unique_cats,
                        help="Select one or more categories to include in calculation"
                    )

                    # Select numeric attribute to sum/average
                    if numeric_cols:
                        value_attr = st.selectbox(
                            "Select Value Attribute:",
                            numeric_cols,
                            help=f"Choose numeric attribute to {'sum' if calc_mode == 'Calculate Block Sum' else 'average'}"
                        )
                    else:
                        st.warning("⚠️ No numeric attributes found in dataset.")

        elif calc_mode in ["Calculate Thickness", "Calculate Stripping Ratio", "Calculate Block Sum", "Calculate Block Average"] and not cat_cols:
            st.warning("⚠️ No categorical attributes found in dataset.")

        # Store configuration in session state
        if 'block_sum_config' not in st.session_state:
            st.session_state.block_sum_config = {}

        st.session_state.block_sum_config = {
            'calc_mode': calc_mode,
            'categorical_attr': categorical_attr,
            'selected_categories': selected_categories,
            'ob_categories': ob_categories,
            'ore_categories': ore_categories,
            'value_attr': value_attr
        }

        # Apply sum if in summed mode (after rerun)
        if st.session_state.is_summed:
            config = st.session_state.block_sum_config
            viz.sum_vertical_blocks(
                categorical_attr=config.get('categorical_attr'),
                calc_mode=CALC_MODE_MAPPING[config.get('calc_mode', 'Calculate Thickness')],
                selected_categories=config.get('selected_categories'),
                ob_categories=config.get('ob_categories'),
                ore_categories=config.get('ore_categories'),
                value_attr=config.get('value_attr')
            )

        # Process Block button
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns([1, 1, 1, 1])

        with col_sum1:
            if st.button("📊 Calculate Block!", use_container_width=True):
                if not st.session_state.is_summed:
                    # Save original if not already saved
                    st.session_state.original_df = viz.df.copy()

                    # Apply vertical sum with configuration
                    config = st.session_state.block_sum_config
                    mode_map = {
                        'Calculate Thickness': 'thickness',
                        'Calculate Stripping Ratio': 'stripping_ratio',
                        'Calculate Block Sum': 'block_sum',
                        'Calculate Block Average': 'block_average'
                    }

                    viz.sum_vertical_blocks(
                        categorical_attr=config.get('categorical_attr'),
                        calc_mode=mode_map[config.get('calc_mode', 'Calculate Thickness')],
                        selected_categories=config.get('selected_categories'),
                        ob_categories=config.get('ob_categories'),
                        ore_categories=config.get('ore_categories'),
                        value_attr=config.get('value_attr')
                    )

                    st.session_state.is_summed = True
                    mode_name = config.get('calc_mode', 'Calculate Thickness')
                    st.success(f"✓ Process applied ({mode_name}): {len(viz.df):,} blocks")
                    st.rerun()  # Rerun to update visualization
                else:
                    st.info("Already in summed mode. Click 'Reset' to restore original.")

        with col_sum2:
            if st.button("🔄 Reset Original", use_container_width=True):
                if st.session_state.is_summed:
                    viz.df = st.session_state.original_df.copy()
                    st.session_state.is_summed = False
                    st.success(f"✓ Original blocks restored: {len(viz.df):,} blocks")
                    st.rerun()  # Rerun to update visualization
                else:
                    st.info("Already showing original data.")

        with col_sum3:
            # Export CSV button with metadata
            csv_data = viz.export_to_csv_with_metadata().encode('utf-8')
            filename_suffix = "summed" if st.session_state.is_summed else "original"
            download_filename = f"block_model_{filename_suffix}_{len(viz.df)}_blocks.csv"

            st.download_button(
                label="💾 Export CSV",
                data=csv_data,
                file_name=download_filename,
                mime="text/csv",
                use_container_width=True,
                help=f"Download {'summed' if st.session_state.is_summed else 'original'} data as CSV (Datamine format with metadata)"
            )

        with col_sum4:
            if st.session_state.is_summed:
                st.info(f"📊 **Summed**: {len(viz.df):,} blocks")
            else:
                st.info(f"📦 **Original**: {len(viz.df):,} blocks")

        # ====================================================================
        # ATTRIBUTE SELECTION
        # ====================================================================
        st.markdown("## 🎯 Attribute Selection")
        st.markdown("---")

        # Get numeric columns
        numeric_cols = viz.df.select_dtypes(include=[np.number]).columns.tolist()
        available_attrs = [col for col in numeric_cols
                          if not any(kw in col.lower() for kw in EXCLUDE_ATTRIBUTE_KEYWORDS)]

        # Get categorical columns
        categorical_cols = viz.df.select_dtypes(include=['object']).columns.tolist()

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
        # FILTERING
        # ====================================================================
        with st.expander("🔍 Data Filtering (Optional)"):
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

                attr_min = float(viz.df[filter_attr].min())
                attr_max = float(viz.df[filter_attr].max())

                if filter_type == "Range":
                    filter_range = st.slider(
                        f"{filter_attr} range:",
                        min_value=attr_min,
                        max_value=attr_max,
                        value=(attr_min, attr_max)
                    )
                    filter_dict = {filter_attr: filter_range}
                elif filter_type == "Minimum value":
                    filter_min = st.slider(
                        f"Minimum {filter_attr}:",
                        min_value=attr_min,
                        max_value=attr_max,
                        value=attr_min
                    )
                    filter_dict = {filter_attr: (filter_min, None)}
                else:
                    filter_max = st.slider(
                        f"Maximum {filter_attr}:",
                        min_value=attr_min,
                        max_value=attr_max,
                        value=attr_max
                    )
                    filter_dict = {filter_attr: (None, filter_max)}

                # Apply filter
                viz.filter_data(filter_dict)
                st.info(f"Filtered to {len(viz.df):,} blocks")

        # ====================================================================
        # AUTO-GENERATE VISUALIZATION
        # ====================================================================
        st.markdown("## 🎨 3D Point Cloud Visualization")
        st.markdown("---")

        if len(selected_attrs) == 0 or color_attr is None:
            st.info("👆 Please select an attribute from the dropdown above to visualize")
        else:
            # Auto-generate visualization
            with st.spinner("🔄 Creating visualization..."):
                # Create point cloud visualization
                fig = viz.visualize_scatter(
                    color_by=color_attr,
                    marker_size=marker_size,
                    opacity=opacity,
                    colorscale=colorscale,
                    title=f"Block Model - {color_attr} ({len(viz.df):,} blocks)"
                )

                # Display
                st.plotly_chart(fig, use_container_width=True)

            # 3D Controls info
            st.markdown(get_3d_controls_html(), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
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

# Footer with modern styling
st.sidebar.markdown("---")

# Read logo and encode to base64
img = Image.open("2.png")
buffered = BytesIO()
img.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Display footer
st.sidebar.markdown(get_footer_html(img_str), unsafe_allow_html=True)