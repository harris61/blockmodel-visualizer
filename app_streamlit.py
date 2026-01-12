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

from block_model_visualizer import BlockModelVisualizer


# Page config
st.set_page_config(
    page_title="Block Model Visualizer",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Header styling */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    h2 {
        color: #1e40af;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3b82f6;
        padding-bottom: 0.5rem;
    }

    h3 {
        color: #2563eb;
        font-weight: 600;
        margin-top: 1rem;
    }

    /* Card-like containers */
    div[data-testid="stExpander"] {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background-color: #f9fafb;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #e0e7ff 100%);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    /* Button styling - apply same gradient style to ALL buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid #000000 !important;
    }

    /* Additional selector for button styling */
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 1px solid #000000 !important;
        box-sizing: border-box;
    }

    /* Info boxes */
    div[data-testid="stMarkdownContainer"] > div > div > div.stAlert {
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Radio buttons */
    div[role="radiogroup"] {
        background-color: #f1f5f9;
        padding: 0.5rem;
        border-radius: 8px;
    }

    /* Select boxes */
    div[data-baseweb="select"] {
        border-radius: 6px;
    }

    /* Multiselect */
    div[data-baseweb="select"] > div {
        border-radius: 6px;
    }

    /* Success message */
    .stSuccess {
        background-color: #d1fae5;
        color: #065f46;
        border-left: 4px solid #10b981;
        border-radius: 6px;
    }

    /* Warning message */
    .stWarning {
        background-color: #fef3c7;
        color: #92400e;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
    }

    /* Hide Streamlit default elements for professional look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Hide rerun/always rerun buttons */
    button[title="Rerun"] {
        display: none;
    }

    div[data-testid="stToolbar"] {
        display: none;
    }

    /* Hide manage app button */
    button[data-testid="manage-app-button"] {display: none;}
    div.stActionButton {display: none;}
</style>
""", unsafe_allow_html=True)

# Title with modern styling
st.markdown("""
<div style='text-align: center; padding: 1.5rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <h1 style='color: white; margin: 0; font-size: 2.5rem; font-weight: 700;'>📦 3D Block Model Visualizer</h1>
    <p style='color: #e0e7ff; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>Interactive web interface untuk visualisasi block model tambang</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for file selection and parameters
st.sidebar.markdown("### 🔧 Settings")
st.sidebar.markdown("---")

# Use point cloud visualization by default (3D boxes removed due to performance)
viz_type = "Point Cloud (Fast)"

# ============================================================================
# FILE SELECTION
# ============================================================================
st.sidebar.markdown("#### 📁 File Upload")

csv_file = None
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV file",
    type=['csv'],
    help="Upload block model CSV file (Datamine, Vulcan, or standard format)"
)

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = Path("temp_uploaded.csv")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    csv_file = str(temp_path)
    st.sidebar.success(f"✓ Loaded: {uploaded_file.name}")

# Skip rows parameter
skip_rows = st.sidebar.number_input(
    "Number of Header Lines",
    min_value=0,
    max_value=10,
    value=3,
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
opacity = 1.0
marker_size = 4

colorscale = st.sidebar.selectbox(
    "🎨 Colorscale:",
    ["Viridis", "Plasma", "Inferno", "Hot", "Portland", "RdYlBu", "Spectral", "Jet"],
    help="Color scheme for visualization (gradient mode)"
)

# Color mode (gradient vs discrete)
st.sidebar.markdown("")
color_mode = st.sidebar.radio(
    "🖌️ Color scheme type:",
    ["Auto-detect", "Gradient (Numeric)", "Discrete (Categorical)"],
    index=0,
    help="Auto-detect: Automatically choose based on data type\n"
         "Gradient: Continuous colorscale for numeric ranges\n"
         "Discrete: Distinct colors for categories"
)

# Map to API parameter
color_mode_map = {
    "Auto-detect": "auto",
    "Gradient (Numeric)": "gradient",
    "Discrete (Categorical)": "discrete"
}
color_mode_param = color_mode_map[color_mode]

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
            ["Calculate Thickness", "Calculate Stripping Ratio", "Calculate Block Sum", "Calculate Block Average"],
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
        exclude_keywords = ['ijk', 'index', 'morig', 'nx', 'ny', 'nz',
                           'centroid_x', 'centroid_y', 'centroid_z',
                           'xc', 'yc', 'zc', 'dim_x', 'dim_y', 'dim_z']

        available_attrs = [col for col in numeric_cols
                          if not any(kw in col.lower() for kw in exclude_keywords)]

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
            st.markdown("""
            <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6; margin-top: 1rem;'>
                <h4 style='color: #1e40af; margin-top: 0;'>🎮 3D Controls</h4>
                <ul style='color: #1e3a8a; margin-bottom: 0;'>
                    <li><strong>Left click + drag</strong>: Rotate</li>
                    <li><strong>Scroll</strong>: Zoom in/out</li>
                    <li><strong>Right click + drag</strong>: Pan (move view)</li>
                    <li><strong>Double click</strong>: Reset camera</li>
                    <li><strong>Hover</strong>: View block details</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())

else:
    # Important Notes
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 6px solid #dc2626; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
        <h2 style='color: #991b1b; margin-top: 0;'>⚠️ Catatan Penting</h2>
        <div style='background: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
            <ol style='color: #374151; margin: 0; line-height: 1.8;'>
                <li><strong>Hanya bisa mengolah Blok Model</strong>, tidak bisa mengolah Stratigraphic Model</li>
                <li><strong>Format file yang didukung:</strong> Blok Model hanya dalam format .csv, baik dalam format Surpac, Datamine, dll</li>
                <li><strong>Visualisasi menggunakan Point Cloud</strong> untuk alasan performa</li>
                <li><strong>Calculate Thickness, Calculate Stripping Ratio, Block Sum, dan Block Average:</strong> attribute categorical yang bisa dibaca hanya yang bertipe data Text atau Categorical</li>
            </ol>
            <div style='background: #fef3c7; padding: 0.75rem; border-radius: 6px; margin-top: 1rem; border-left: 4px solid #f59e0b;'>
                <p style='color: #78350f; margin: 0;'><strong>💡 Tips:</strong> Buat attribute di dalam block model berupa klasifikasi material dengan tipe data teks terlebih dahulu</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # User guide with modern styling
    st.markdown("""
    <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 6px solid #f59e0b;'>
        <h2 style='color: #92400e; margin-top: 0;'>📖 Panduan Penggunaan</h2>
        <p style='color: #78350f; margin-bottom: 0;'>Ikuti langkah-langkah berikut untuk memulai visualisasi block model Anda</p>
    </div>
    """, unsafe_allow_html=True)

    # Step 1
    st.markdown("""
    <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem; border-left: 5px solid #667eea;'>
        <h3 style='color: #667eea; margin-top: 0;'>1️⃣ Upload & Setup</h3>
        <ul style='color: #4b5563;'>
            <li>Upload file CSV (Surpac/Datamine/format standar)</li>
            <li>Atur "Number of Header Lines"</li>
            <li>Pilih <strong>Colorscale</strong> dan <strong>Color Mode</strong> (Auto/Gradient/Discrete)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Step 2
    st.markdown("""
    <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem; border-left: 5px solid #10b981;'>
        <h3 style='color: #10b981; margin-top: 0;'>2️⃣ Block Sum Configuration</h3>
        <p style='color: #4b5563; font-weight: 600; margin-top: 1rem;'>Mode Calculate Thickness:</p>
        <ul style='color: #6b7280; margin-top: 0.5rem;'>
            <li>Pilih categorical attribute (lithology/zone)</li>
            <li>Pilih kategori yang ingin dihitung thickness-nya</li>
            <li>Hasil: kolom baru <code>thickness_{kategori}</code></li>
        </ul>
        <p style='color: #4b5563; font-weight: 600; margin-top: 1rem;'>Mode Calculate Stripping Ratio:</p>
        <ul style='color: #6b7280; margin-top: 0.5rem;'>
            <li>Tentukan kategori OB/Waste dan Ore</li>
            <li>Hasil: <code>thickness_OB</code>, <code>thickness_Ore</code>, <code>stripping_ratio</code></li>
            <li>SR = thickness_OB / thickness_Ore per kolom</li>
        </ul>
        <p style='color: #4b5563; font-weight: 600; margin-top: 1rem;'>Mode Calculate Block Sum:</p>
        <ul style='color: #6b7280; margin-top: 0.5rem;'>
            <li>Pilih categorical attribute (misalnya: material_type)</li>
            <li>Pilih kategori yang ingin dihitung (misalnya: ore, waste)</li>
            <li>Pilih value attribute yang ingin dijumlahkan (misalnya: grade, tonnage)</li>
            <li>Hasil: kolom baru <code>sum_{kategori}_{attribute}</code> untuk setiap kategori</li>
        </ul>
        <p style='color: #4b5563; font-weight: 600; margin-top: 1rem;'>Mode Calculate Block Average:</p>
        <ul style='color: #6b7280; margin-top: 0.5rem;'>
            <li>Pilih categorical attribute (misalnya: material_type)</li>
            <li>Pilih kategori yang ingin dihitung (misalnya: ore, waste)</li>
            <li>Pilih value attribute yang ingin dirata-rata (misalnya: grade, tonnage)</li>
            <li>Hasil: kolom baru <code>avg_{kategori}_{attribute}</code> untuk setiap kategori</li>
        </ul>
        <p style='color: #059669; font-weight: 600; margin-top: 1rem;'>✓ Klik "Calculate Block!" → Reset/Export tersedia</p>
    </div>
    """, unsafe_allow_html=True)

    # Step 3
    st.markdown("""
    <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem; border-left: 5px solid #f59e0b;'>
        <h3 style='color: #f59e0b; margin-top: 0;'>3️⃣ Visualisasi Point Cloud</h3>
        <ul style='color: #4b5563;'>
            <li>Pilih attribute dari <strong>Numeric</strong> (grade, tonnage, SR) atau <strong>Categorical</strong> (lithology, zone)</li>
            <li>Visualisasi 3D otomatis muncul</li>
            <li>Kontrol: <strong>Drag</strong> (rotasi) | <strong>Scroll</strong> (zoom) | <strong>Right-drag</strong> (pan) | <strong>Double-click</strong> (reset)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Step 4
    st.markdown("""
    <div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem; border-left: 5px solid #8b5cf6;'>
        <h3 style='color: #8b5cf6; margin-top: 0;'>4️⃣ Filter & Export</h3>
        <ul style='color: #4b5563;'>
            <li>Filter data berdasarkan range attribute</li>
            <li>Export hasil dengan <strong>"Export CSV"</strong> (termasuk kolom thickness/SR)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer with modern styling
st.sidebar.markdown("---")

# Read logo and encode to base64
from PIL import Image
import base64
from io import BytesIO

# Read image and convert to base64
img = Image.open("2.png")
buffered = BytesIO()
img.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

# Single container with everything
footer_html = f"""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 8px; text-align: center;'>
    <p style='color: white; font-weight: 700; font-size: 1.1rem; margin: 0;'>Block Model Visualizer</p>
    <p style='color: #e0e7ff; font-size: 0.9rem; margin: 0.5rem 0 1rem 0;'>v1.0</p>
    <div style='border-top: 1px solid rgba(255,255,255,0.3); padding-top: 1rem; padding-bottom: 1rem;'>
        <p style='color: #e0e7ff; font-size: 0.8rem; margin: 0.3rem 0 0.2rem 0;'>Created by:</p>
        <a href='https://www.linkedin.com/in/harristio-adam/' target='_blank' style='color: white; text-decoration: none; font-weight: 600; font-size: 0.85rem;'>Harristio Adam</a>
        <p style='color: #e0e7ff; font-size: 0.8rem; margin: 0.8rem 0 0.3rem 0;'>Powered by:</p>
        <a href='https://www.linkedin.com/company/soft-roc' target='_blank'>
            <img src='data:image/png;base64,{img_str}' style='max-width: 100px; height: auto;' alt='Soft.Roc'>
        </a>
    </div>
</div>
"""

st.sidebar.markdown(footer_html, unsafe_allow_html=True)