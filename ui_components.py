"""
UI Components for Block Model Visualizer

This module contains reusable UI components for the Streamlit app,
particularly the Block Sum configuration section.
"""

import streamlit as st
import numpy as np
from typing import Dict, List, Optional, Any
from block_model_visualizer import BlockModelVisualizer
from config import CALC_MODE_OPTIONS, CALC_MODE_MAPPING


def initialize_session_state(viz: BlockModelVisualizer) -> None:
    """
    Initialize session state for block sum operations

    Args:
        viz: BlockModelVisualizer instance with loaded data
    """
    if 'original_df' not in st.session_state:
        st.session_state.original_df = viz.df.copy()
        st.session_state.is_summed = False

    if 'block_sum_config' not in st.session_state:
        st.session_state.block_sum_config = {}


def render_block_sum_config(viz: BlockModelVisualizer) -> Dict[str, Any]:
    """
    Render the Block Sum configuration UI section

    Args:
        viz: BlockModelVisualizer instance with loaded data

    Returns:
        dict: Configuration dictionary with user selections
    """
    st.markdown("## Block Sum Configuration")
    st.markdown("---")

    # Calculation Mode selection
    calc_mode = st.radio(
        "Calculation Mode:",
        CALC_MODE_OPTIONS,
        horizontal=False,
        help="Thickness: Calculate thickness by category | SR: OB/Ore ratio | "
             "Block Sum: Sum attribute values by category | Block Average: Average attribute values by category"
    )

    # Get categorical and numeric columns
    cat_cols = viz.df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    numeric_cols = viz.df.select_dtypes(include=[np.number]).columns.tolist()

    # Initialize config variables
    categorical_attr = None
    selected_categories = None
    ob_categories = None
    ore_categories = None
    value_attr = None

    # Render appropriate UI based on calc mode and available columns
    if calc_mode in CALC_MODE_OPTIONS and cat_cols:
        categorical_attr = st.selectbox(
            "Select Categorical Attribute:",
            cat_cols,
            help="Choose attribute for classification (e.g., lithology, zone, rock_type)"
        )

        if categorical_attr:
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
                selected_categories = st.multiselect(
                    "Select Categories to Calculate:",
                    unique_cats,
                    default=unique_cats[:2] if len(unique_cats) >= 2 else unique_cats,
                    help="Select one or more categories to include in calculation"
                )

                if numeric_cols:
                    action_verb = 'sum' if calc_mode == 'Calculate Block Sum' else 'average'
                    value_attr = st.selectbox(
                        "Select Value Attribute:",
                        numeric_cols,
                        help=f"Choose numeric attribute to {action_verb}"
                    )
                else:
                    st.warning("No numeric attributes found in dataset.")

    elif calc_mode in CALC_MODE_OPTIONS and not cat_cols:
        st.warning("No categorical attributes found in dataset.")

    # Build and store configuration
    config = {
        'calc_mode': calc_mode,
        'categorical_attr': categorical_attr,
        'selected_categories': selected_categories,
        'ob_categories': ob_categories,
        'ore_categories': ore_categories,
        'value_attr': value_attr
    }

    st.session_state.block_sum_config = config

    return config


def apply_vertical_sum(viz: BlockModelVisualizer, config: Dict[str, Any]) -> None:
    """
    Apply vertical block summing based on configuration

    Args:
        viz: BlockModelVisualizer instance
        config: Configuration dictionary
    """
    calc_mode_key = config.get('calc_mode', 'Calculate Thickness')
    calc_mode_param = CALC_MODE_MAPPING[calc_mode_key]

    viz.sum_vertical_blocks(
        categorical_attr=config.get('categorical_attr'),
        calc_mode=calc_mode_param,
        selected_categories=config.get('selected_categories'),
        ob_categories=config.get('ob_categories'),
        ore_categories=config.get('ore_categories'),
        value_attr=config.get('value_attr')
    )


def render_block_sum_controls(viz: BlockModelVisualizer) -> Optional[str]:
    """
    Render the 4 control buttons for block sum operations

    Args:
        viz: BlockModelVisualizer instance

    Returns:
        str or None: Action taken ('calculate', 'reset') or None
    """
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns([1, 1, 1, 1])

    action = None

    # Calculate Button
    with col_sum1:
        if st.button("Calculate Block!", use_container_width=True):
            if not st.session_state.is_summed:
                st.session_state.original_df = viz.df.copy()

                config = st.session_state.block_sum_config
                apply_vertical_sum(viz, config)

                st.session_state.is_summed = True
                mode_name = config.get('calc_mode', 'Calculate Thickness')
                st.success(f"Process applied ({mode_name}): {len(viz.df):,} blocks")
                action = 'calculate'
            else:
                st.info("Already in summed mode. Click 'Reset' to restore original.")

    # Reset Button
    with col_sum2:
        if st.button("Reset Original", use_container_width=True):
            if st.session_state.is_summed:
                viz.df = st.session_state.original_df.copy()
                st.session_state.is_summed = False
                # Clear any active filters when resetting
                if 'filter_dict' in st.session_state:
                    st.session_state.filter_dict = None
                if 'filter_applied' in st.session_state:
                    st.session_state.filter_applied = False
                st.success(f"Original blocks restored: {len(viz.df):,} blocks")
                action = 'reset'
            else:
                st.info("Already showing original data.")

    # Export Button
    with col_sum3:
        csv_data = viz.export_to_csv_with_metadata().encode('utf-8')
        filename_suffix = "summed" if st.session_state.is_summed else "original"
        download_filename = f"block_model_{filename_suffix}_{len(viz.df)}_blocks.csv"

        st.download_button(
            label="Export CSV",
            data=csv_data,
            file_name=download_filename,
            mime="text/csv",
            use_container_width=True,
            help=f"Download {'summed' if st.session_state.is_summed else 'original'} data as CSV (Datamine format with metadata)"
        )

    # Status Info
    with col_sum4:
        if st.session_state.is_summed:
            st.info(f"**Summed**: {len(viz.df):,} blocks")
        else:
            st.info(f"**Original**: {len(viz.df):,} blocks")

    return action
