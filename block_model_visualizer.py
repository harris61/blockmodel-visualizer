"""
3D Block Model Visualizer with Multiple Visualization Modes

Features:
- Point cloud visualization (fast, for large datasets)
- 3D box/cube rendering with actual block dimensions (detailed)
- Interactive dropdown menu to change visualization attribute
- Block summing along Z-axis
- Data filtering and sampling
- CSV export with metadata preservation
- Format-agnostic (Vulcan, Datamine, MineSight, etc.)
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pathlib import Path
import argparse


class BlockModelVisualizer:
    """
    Unified block model visualizer with scatter and box visualization modes
    """

    def __init__(self, csv_file, skip_rows=3):
        """
        Initialize the advanced visualizer

        Args:
            csv_file (str): Path to the CSV file
            skip_rows (int): Number of metadata rows to skip
        """
        self.csv_file = Path(csv_file)
        self.skip_rows = skip_rows
        self.df = None
        self.coord_cols = None
        self.dim_cols = None
        self.metadata_lines = []  # Store original metadata lines

        if not self.csv_file.exists():
            raise FileNotFoundError(f"File not found: {csv_file}")

    def load_data(self):
        """Load CSV data"""
        print(f"Loading data from {self.csv_file}...")

        if self.skip_rows > 0:
            # Store metadata lines for later export
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                lines = [next(f).strip() for _ in range(self.skip_rows + 1)]
                self.metadata_lines = lines[1:]  # Skip header, store metadata rows

            self.df = pd.read_csv(self.csv_file, skiprows=range(1, self.skip_rows + 1))
            print(f"Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")
            print(f"Stored {len(self.metadata_lines)} metadata lines for export")
        else:
            self.df = pd.read_csv(self.csv_file)
            print(f"Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")

        self._detect_columns()
        return self

    def _detect_columns(self):
        """Detect coordinate and dimension columns"""
        coord_patterns = {
            'x': ['centroid_x', 'xc', 'x_coord', 'x', 'xcenter'],
            'y': ['centroid_y', 'yc', 'y_coord', 'y', 'ycenter'],
            'z': ['centroid_z', 'zc', 'z_coord', 'z', 'zcenter', 'elevation']
        }

        dim_patterns = {
            'dx': ['dim_x', 'xinc', 'dx', 'x_size'],
            'dy': ['dim_y', 'yinc', 'dy', 'y_size'],
            'dz': ['dim_z', 'zinc', 'dz', 'z_size']
        }

        self.coord_cols = {}
        self.dim_cols = {}

        for axis, patterns in coord_patterns.items():
            for pattern in patterns:
                matching = [col for col in self.df.columns if pattern.lower() in col.lower()]
                if matching:
                    self.coord_cols[axis] = matching[0]
                    break

        for axis, patterns in dim_patterns.items():
            for pattern in patterns:
                matching = [col for col in self.df.columns if pattern.lower() in col.lower()]
                if matching:
                    self.dim_cols[axis] = matching[0]
                    break

        if len(self.coord_cols) != 3:
            raise ValueError(f"Could not detect all coordinate columns. Found: {self.coord_cols}")

        print(f"Coordinates: {self.coord_cols}")
        print(f"Dimensions: {self.dim_cols}")

    def sample_data(self, n_samples=None, method='stratified_z'):
        """
        Sample data for performance

        Args:
            n_samples (int): Number of samples
            method (str): 'random', 'spatial', or 'stratified_z'
                - random: Pure random sampling (old method, may create Z gaps)
                - spatial: Grid-based spatial sampling (better 3D continuity)
                - stratified_z: Stratified by Z-level (ensures Z coverage)
        """
        if n_samples is None or n_samples >= len(self.df):
            return self

        print(f"Sampling {n_samples} blocks from {len(self.df)} using '{method}' method...")

        if method == 'random':
            # Old method - pure random
            self.df = self.df.sample(n=n_samples, random_state=42)

        elif method == 'stratified_z':
            # Stratified sampling by Z-level to ensure vertical continuity
            z_col = self.coord_cols['z']
            unique_z = sorted(self.df[z_col].unique())
            n_z_levels = len(unique_z)

            # Calculate samples per Z-level
            samples_per_z = max(1, n_samples // n_z_levels)

            sampled_dfs = []
            for z_val in unique_z:
                z_blocks = self.df[self.df[z_col] == z_val]
                n_take = min(samples_per_z, len(z_blocks))
                if n_take > 0:
                    sampled_dfs.append(z_blocks.sample(n=n_take, random_state=42))

            self.df = pd.concat(sampled_dfs, ignore_index=True)

            # If we have more samples than needed, randomly remove some
            if len(self.df) > n_samples:
                self.df = self.df.sample(n=n_samples, random_state=42)

            print(f"  > Sampled from {n_z_levels} Z-levels with ~{samples_per_z} blocks/level")

        elif method == 'spatial':
            # Spatial grid-based sampling for better 3D coverage
            x_col = self.coord_cols['x']
            y_col = self.coord_cols['y']
            z_col = self.coord_cols['z']

            # Create 3D grid bins
            n_bins = int(np.cbrt(n_samples * 2))  # Approximate cube root

            self.df['_x_bin'] = pd.cut(self.df[x_col], bins=n_bins, labels=False)
            self.df['_y_bin'] = pd.cut(self.df[y_col], bins=n_bins, labels=False)
            self.df['_z_bin'] = pd.cut(self.df[z_col], bins=n_bins, labels=False)

            # Sample from each grid cell
            sampled_dfs = []
            for (x_bin, y_bin, z_bin), group in self.df.groupby(['_x_bin', '_y_bin', '_z_bin']):
                n_take = max(1, int(n_samples / (n_bins**3)))
                n_take = min(n_take, len(group))
                if n_take > 0:
                    sampled_dfs.append(group.sample(n=n_take, random_state=42))

            self.df = pd.concat(sampled_dfs, ignore_index=True)

            # Remove temporary columns
            self.df = self.df.drop(columns=['_x_bin', '_y_bin', '_z_bin'])

            # Adjust to target size
            if len(self.df) > n_samples:
                self.df = self.df.sample(n=n_samples, random_state=42)

            print(f"  > Spatial grid sampling with {n_bins}x{n_bins}x{n_bins} bins")

        print(f"Sampled to {len(self.df)} blocks")
        return self

    def filter_data(self, filters=None):
        """Filter data based on conditions"""
        if filters is None:
            return self

        filtered_df = self.df.copy()

        for col, condition in filters.items():
            if col not in filtered_df.columns:
                print(f"Warning: Column '{col}' not found")
                continue

            if isinstance(condition, tuple):
                min_val, max_val = condition
                if min_val is not None:
                    filtered_df = filtered_df[filtered_df[col] >= min_val]
                if max_val is not None:
                    filtered_df = filtered_df[filtered_df[col] <= max_val]
            elif isinstance(condition, list):
                filtered_df = filtered_df[filtered_df[col].isin(condition)]

        print(f"Filtered from {len(self.df)} to {len(filtered_df)} blocks")
        self.df = filtered_df
        return self

    def get_numeric_columns(self):
        """
        Get list of numeric columns suitable for visualization (excludes structural columns)
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude structural/grid columns
        exclude_keywords = ['ijk', 'index', 'morig', 'nx', 'ny', 'nz']
        filtered_cols = [col for col in numeric_cols
                        if not any(keyword in col.lower() for keyword in exclude_keywords)]
        return filtered_cols

    def get_categorical_columns(self):
        """
        Get list of categorical columns suitable for visualization
        """
        cat_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        return cat_cols

    def sum_vertical_blocks(self, categorical_attr=None, calc_mode='all',
                            selected_categories=None, ob_categories=None, ore_categories=None,
                            value_attr=None):
        """
        Sum blocks vertically (along Z-axis) for each X,Y coordinate

        For each unique (X,Y) position:
        - Sum all numeric attributes across all Z levels
        - Place result at highest Z level
        - Remove other blocks at that X,Y position

        Enhanced modes:
        - 'all': Sum all blocks (default, original behavior)
        - 'thickness': Calculate thickness for selected categories
        - 'stripping_ratio': Calculate OB/Ore thickness and SR
        - 'block_sum': Sum value attribute for selected categories
        - 'block_average': Average value attribute for selected categories

        Args:
            categorical_attr (str): Name of categorical attribute for classification
            calc_mode (str): 'all', 'thickness', 'stripping_ratio', 'block_sum', or 'block_average'
            selected_categories (list): Categories to calculate thickness/sum/average for
            ob_categories (list): Categories considered as Overburden/Waste
            ore_categories (list): Categories considered as Ore
            value_attr (str): Name of numeric attribute to sum/average (for block_sum/block_average mode)

        Returns:
            self (for method chaining)
        """
        if self.df is None or len(self.df) == 0:
            print("Warning: No data to sum")
            return self

        print(f"\nSumming vertical blocks (collapsing Z-axis)...")
        print(f"Calculation mode: {calc_mode}")
        if calc_mode != 'all':
            print(f"Categorical attribute: {categorical_attr}")
            if calc_mode == 'thickness':
                print(f"Selected categories: {selected_categories}")
            elif calc_mode == 'stripping_ratio':
                print(f"OB categories: {ob_categories}")
                print(f"Ore categories: {ore_categories}")
            elif calc_mode in ['block_sum', 'block_average']:
                print(f"Selected categories: {selected_categories}")
                print(f"Value attribute: {value_attr}")
        print(f"Original blocks: {len(self.df):,}")

        x_col = self.coord_cols['x']
        y_col = self.coord_cols['y']
        z_col = self.coord_cols['z']

        # Get numeric columns to sum (exclude coordinates and dimensions)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude structural/geometric attributes that should NOT be summed
        exclude_cols = [
            x_col, y_col, z_col,  # Main coordinates
            'ijk',                 # Block index
            'xc', 'yc', 'zc',     # Coordinate fields
            'xinc', 'yinc', 'zinc',  # Block increments/sizes
            'xmorig', 'ymorig', 'zmorig',  # Grid origin
            'nx', 'ny', 'nz',     # Grid dimensions
            'fillvol',            # Fill volume
            'volume',             # Total volume
            'index', 'morig'      # Other indices
        ]

        # Also exclude dimension columns
        if self.dim_cols:
            exclude_cols.extend(self.dim_cols.values())

        # Remove duplicates and filter only existing columns
        exclude_cols = list(set([col for col in exclude_cols if col in numeric_cols]))
        sum_cols = [col for col in numeric_cols if col not in exclude_cols]

        # Get categorical/non-numeric columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        print(f"Grouping by: ({x_col}, {y_col})")
        print(f"Excluded (preserved) attributes: {sorted(exclude_cols)}")
        print(f"Summing attributes: {sorted(sum_cols)}")

        # Group by X,Y and aggregate
        grouped = self.df.groupby([x_col, y_col])

        summed_blocks = []

        # Get block height (dz) if available
        dz_col = self.dim_cols.get('dz') if self.dim_cols else None

        for (x, y), group in grouped:
            # Get highest Z level for this X,Y position
            max_z_idx = group[z_col].idxmax()
            max_z_row = group.loc[max_z_idx].copy()

            # Sum numeric attributes
            for col in sum_cols:
                if col in group.columns:
                    max_z_row[col] = group[col].sum()

            # For categorical columns, take the most frequent value
            for col in categorical_cols:
                if col in group.columns:
                    max_z_row[col] = group[col].mode()[0] if len(group[col].mode()) > 0 else group[col].iloc[0]

            # === ENHANCED CALCULATIONS ===
            if calc_mode == 'thickness' and categorical_attr and selected_categories:
                # Calculate thickness for each selected category
                for category in selected_categories:
                    category_blocks = group[group[categorical_attr] == category]
                    if dz_col and dz_col in group.columns:
                        thickness = category_blocks[dz_col].sum()
                    else:
                        # Fallback: count blocks (assume uniform height)
                        thickness = len(category_blocks)

                    col_name = f'thickness_{category}'
                    max_z_row[col_name] = thickness

            elif calc_mode == 'stripping_ratio' and categorical_attr and ob_categories and ore_categories:
                # Calculate OB thickness
                ob_blocks = group[group[categorical_attr].isin(ob_categories)]
                if dz_col and dz_col in group.columns:
                    thickness_ob = ob_blocks[dz_col].sum()
                else:
                    thickness_ob = len(ob_blocks)

                # Calculate Ore thickness
                ore_blocks = group[group[categorical_attr].isin(ore_categories)]
                if dz_col and dz_col in group.columns:
                    thickness_ore = ore_blocks[dz_col].sum()
                else:
                    thickness_ore = len(ore_blocks)

                # Calculate Stripping Ratio
                if thickness_ore > 0:
                    sr = thickness_ob / thickness_ore
                else:
                    sr = np.nan  # or np.inf

                max_z_row['thickness_OB'] = thickness_ob
                max_z_row['thickness_Ore'] = thickness_ore
                max_z_row['stripping_ratio'] = sr

            elif calc_mode == 'block_sum' and categorical_attr and selected_categories and value_attr:
                # Sum value attribute for selected categories
                for category in selected_categories:
                    category_blocks = group[group[categorical_attr] == category]
                    if value_attr in group.columns:
                        sum_value = category_blocks[value_attr].sum()
                    else:
                        sum_value = 0

                    col_name = f'sum_{category}_{value_attr}'
                    max_z_row[col_name] = sum_value

            elif calc_mode == 'block_average' and categorical_attr and selected_categories and value_attr:
                # Average value attribute for selected categories
                for category in selected_categories:
                    category_blocks = group[group[categorical_attr] == category]
                    if value_attr in group.columns and len(category_blocks) > 0:
                        avg_value = category_blocks[value_attr].mean()
                    else:
                        avg_value = np.nan

                    col_name = f'avg_{category}_{value_attr}'
                    max_z_row[col_name] = avg_value

            # Keep dimensions from the top block (maintain grid spacing for Vulcan compatibility)
            # dim_z should remain as grid spacing (zinc) - do NOT sum it
            # This ensures blocks can be placed in Vulcan's regular grid structure

            summed_blocks.append(max_z_row)

        # Create new dataframe
        self.df = pd.DataFrame(summed_blocks).reset_index(drop=True)

        print(f"Summed blocks: {len(self.df):,}")
        print(f"Reduction: {len(self.df.groupby([x_col, y_col]))} unique (X,Y) positions")
        print(f"Each position now has 1 block at highest Z level")

        # Print summary for thickness/SR/block_sum/block_average calculations
        if calc_mode == 'thickness' and selected_categories:
            print("\n--- Thickness Calculation Summary ---")
            for category in selected_categories:
                col_name = f'thickness_{category}'
                if col_name in self.df.columns:
                    total = self.df[col_name].sum()
                    avg = self.df[col_name].mean()
                    print(f"{category}: Total = {total:.2f}, Average = {avg:.2f}")

        elif calc_mode == 'stripping_ratio':
            print("\n--- Stripping Ratio Summary ---")
            if 'thickness_OB' in self.df.columns:
                print(f"Total OB Thickness: {self.df['thickness_OB'].sum():.2f}")
                print(f"Average OB Thickness: {self.df['thickness_OB'].mean():.2f}")
            if 'thickness_Ore' in self.df.columns:
                print(f"Total Ore Thickness: {self.df['thickness_Ore'].sum():.2f}")
                print(f"Average Ore Thickness: {self.df['thickness_Ore'].mean():.2f}")
            if 'stripping_ratio' in self.df.columns:
                sr_mean = self.df['stripping_ratio'].mean()
                sr_median = self.df['stripping_ratio'].median()
                print(f"Average SR: {sr_mean:.2f}")
                print(f"Median SR: {sr_median:.2f}")

        elif calc_mode == 'block_sum' and selected_categories and value_attr:
            print(f"\n--- Block Sum Calculation Summary ({value_attr}) ---")
            for category in selected_categories:
                col_name = f'sum_{category}_{value_attr}'
                if col_name in self.df.columns:
                    total = self.df[col_name].sum()
                    avg = self.df[col_name].mean()
                    max_val = self.df[col_name].max()
                    min_val = self.df[col_name].min()
                    print(f"{category}: Total = {total:.2f}, Average = {avg:.2f}, Max = {max_val:.2f}, Min = {min_val:.2f}")

        elif calc_mode == 'block_average' and selected_categories and value_attr:
            print(f"\n--- Block Average Calculation Summary ({value_attr}) ---")
            for category in selected_categories:
                col_name = f'avg_{category}_{value_attr}'
                if col_name in self.df.columns:
                    overall_avg = self.df[col_name].mean()
                    max_val = self.df[col_name].max()
                    min_val = self.df[col_name].min()
                    print(f"{category}: Overall Average = {overall_avg:.2f}, Max = {max_val:.2f}, Min = {min_val:.2f}")

        return self

    def export_to_csv_with_metadata(self):
        """
        Export current dataframe to CSV with Datamine metadata format

        Generic export that preserves:
        - All columns from input file (in their original order)
        - All metadata rows
        - Block model structure as-is

        Does not assume any specific software format (Vulcan, Datamine, etc.)

        Returns:
            str: CSV content with metadata as string
        """
        import io

        # Get current column names (preserve whatever columns exist)
        current_cols = list(self.df.columns)

        # Build CSV content
        output = io.StringIO()

        if self.metadata_lines and len(self.metadata_lines) == 3:
            # Parse original metadata to get column mapping
            # Line 0: Variable descriptions
            # Line 1: Variable types
            # Line 2: Variable defaults

            desc_parts = self.metadata_lines[0].split(',')
            type_parts = self.metadata_lines[1].split(',')
            default_parts = self.metadata_lines[2].split(',')

            # Read original header to map columns
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                original_header = f.readline().strip().split(',')

            # Create column index mapping
            col_to_idx = {col: idx for idx, col in enumerate(original_header)}

            # Build metadata for current columns
            new_desc = []
            new_types = []
            new_defaults = []

            for col in current_cols:
                if col in col_to_idx:
                    idx = col_to_idx[col]
                    new_desc.append(desc_parts[idx] if idx < len(desc_parts) else '')
                    new_types.append(type_parts[idx] if idx < len(type_parts) else 'float')
                    new_defaults.append(default_parts[idx] if idx < len(default_parts) else '0.0')
                else:
                    # New column (e.g., from aggregation)
                    new_desc.append(f'Datamine field : {col}')
                    new_types.append('float')
                    new_defaults.append('0.0')

            # Write header
            output.write(','.join(current_cols) + '\n')

            # Write metadata
            output.write(','.join(new_desc) + '\n')
            output.write(','.join(new_types) + '\n')
            output.write(','.join(new_defaults) + '\n')

        else:
            # No metadata available, write simple header
            output.write(','.join(current_cols) + '\n')

        # Write data with 6 decimal places (Datamine/Vulcan format requirement)
        self.df.to_csv(output, index=False, header=False, float_format='%.6f')

        return output.getvalue()

    def _create_box_mesh(self, x, y, z, dx, dy, dz, color_value, colorscale, vmin, vmax):
        """
        Create a 3D box mesh for a single block

        Returns vertices and faces for the box
        """
        # Define 8 vertices of the box
        vertices = np.array([
            [x - dx/2, y - dy/2, z - dz/2],  # 0: bottom-front-left
            [x + dx/2, y - dy/2, z - dz/2],  # 1: bottom-front-right
            [x + dx/2, y + dy/2, z - dz/2],  # 2: bottom-back-right
            [x - dx/2, y + dy/2, z - dz/2],  # 3: bottom-back-left
            [x - dx/2, y - dy/2, z + dz/2],  # 4: top-front-left
            [x + dx/2, y - dy/2, z + dz/2],  # 5: top-front-right
            [x + dx/2, y + dy/2, z + dz/2],  # 6: top-back-right
            [x - dx/2, y + dy/2, z + dz/2],  # 7: top-back-left
        ])

        # Define 12 triangular faces (2 per box face)
        faces = np.array([
            # Bottom face (z-)
            [0, 1, 2], [0, 2, 3],
            # Top face (z+)
            [4, 6, 5], [4, 7, 6],
            # Front face (y-)
            [0, 5, 1], [0, 4, 5],
            # Back face (y+)
            [2, 7, 3], [2, 6, 7],
            # Left face (x-)
            [0, 3, 7], [0, 7, 4],
            # Right face (x+)
            [1, 5, 6], [1, 6, 2],
        ])

        return vertices, faces

    def _get_discrete_colorscale(self, categories):
        """
        Generate discrete colorscale for categorical data

        Args:
            categories (list): List of unique categories

        Returns:
            list: Discrete colorscale mapping
        """
        # Predefined color palette for discrete categories
        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]

        # Create mapping from category to color
        n_categories = len(categories)
        category_colors = {}

        for i, cat in enumerate(categories):
            category_colors[cat] = colors[i % len(colors)]

        return category_colors

    def _detect_attribute_type(self, attr, df_viz):
        """
        Detect if attribute is numeric (continuous) or categorical (discrete)

        Args:
            attr (str): Attribute name
            df_viz (DataFrame): Data

        Returns:
            str: 'numeric' or 'categorical'
        """
        if pd.api.types.is_numeric_dtype(df_viz[attr]):
            # Check if it's actually discrete numeric (like material codes)
            n_unique = df_viz[attr].nunique()
            n_total = len(df_viz)

            # If less than 20 unique values and less than 10% of total, treat as categorical
            if n_unique <= 20 and n_unique / n_total < 0.1:
                return 'categorical'
            else:
                return 'numeric'
        else:
            # String/object type = categorical
            return 'categorical'

    def create_box_visualization(self, attributes=['ni', 'co', 'fe'],
                                  colorscale='Viridis', opacity=0.9,
                                  title=None, max_blocks=None,
                                  color_mode='auto'):
        """
        Create 3D box visualization with interactive attribute menu

        Args:
            attributes (list): List of attribute names to include in dropdown
            colorscale (str): Colorscale name (for gradient mode)
            opacity (float): Box opacity
            title (str): Plot title
            max_blocks (int): Optional limit for performance (None = all blocks)
            color_mode (str): 'auto', 'gradient', or 'discrete'
                - auto: Auto-detect based on data type
                - gradient: Force gradient colorscale (for numeric ranges)
                - discrete: Force discrete colorscale (for categories)

        Returns:
            plotly.graph_objects.Figure
        """
        # Use all blocks by default
        df_viz = self.df

        # Optional limiting if user specifies max_blocks
        if max_blocks is not None and len(self.df) > max_blocks:
            print(f"Warning: Limiting to {max_blocks} blocks (original: {len(self.df)})")
            print(f"  Set max_blocks=None to render all blocks")
            df_viz = self.df.head(max_blocks)

        # Performance warning for large datasets
        if len(df_viz) > 5000:
            print(f"Warning: Rendering {len(df_viz)} blocks may be slow")
            print(f"  Consider using sample_data() first or set max_blocks parameter")

        # Get numeric columns if attributes not specified
        if not attributes or len(attributes) == 0:
            numeric_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
            exclude = ['ijk', 'index', 'morig', 'nx', 'ny', 'nz', 'centroid_x', 'centroid_y', 'centroid_z']
            attributes = [col for col in numeric_cols if not any(ex in col.lower() for ex in exclude)][:10]

        print(f"Creating box visualization for {len(df_viz)} blocks...")
        print(f"Attributes available: {attributes}")

        # Create traces for each attribute
        traces = []

        for attr_idx, attr in enumerate(attributes):
            if attr not in df_viz.columns:
                print(f"Warning: Attribute '{attr}' not found, skipping")
                continue

            # Detect attribute type
            if color_mode == 'auto':
                attr_type = self._detect_attribute_type(attr, df_viz)
            elif color_mode == 'gradient':
                attr_type = 'numeric'
            else:  # discrete
                attr_type = 'categorical'

            print(f"  - {attr}: {attr_type} ({'gradient' if attr_type == 'numeric' else 'discrete'} colorscale)")

            # Prepare color mapping based on type
            if attr_type == 'categorical':
                # Get unique categories
                unique_vals = sorted(df_viz[attr].dropna().unique())
                category_colors = self._get_discrete_colorscale(unique_vals)

                # Convert categories to numeric indices for colorscale
                category_to_idx = {cat: idx for idx, cat in enumerate(unique_vals)}

                # Create custom discrete colorscale for plotly
                # Format: [[0, color1], [0.33, color1], [0.33, color2], [0.66, color2], ...]
                n_cats = len(unique_vals)
                plotly_colorscale = []
                for i, cat in enumerate(unique_vals):
                    pos_start = i / n_cats
                    pos_end = (i + 1) / n_cats
                    color = category_colors[cat]
                    plotly_colorscale.append([pos_start, color])
                    plotly_colorscale.append([pos_end, color])

                vmin = 0
                vmax = n_cats - 1
            else:
                # Numeric: use gradient colorscale
                color_values = df_viz[attr].values
                vmin = np.nanmin(color_values)
                vmax = np.nanmax(color_values)
                plotly_colorscale = colorscale
                category_to_idx = None

            # Collect all vertices and faces
            all_x, all_y, all_z = [], [], []
            all_i, all_j, all_k = [], [], []
            all_colors = []
            all_hover_texts = []

            vertex_offset = 0

            for idx, row in df_viz.iterrows():
                x = row[self.coord_cols['x']]
                y = row[self.coord_cols['y']]
                z = row[self.coord_cols['z']]

                # Get dimensions
                if self.dim_cols:
                    dx = row.get(self.dim_cols.get('dx', 'dim_x'), 12.5)
                    dy = row.get(self.dim_cols.get('dy', 'dim_y'), 12.5)
                    dz = row.get(self.dim_cols.get('dz', 'dim_z'), 1.0)
                else:
                    dx, dy, dz = 12.5, 12.5, 1.0

                raw_val = row[attr]

                # Convert to color value
                if attr_type == 'categorical':
                    color_val = category_to_idx.get(raw_val, 0)
                    hover_text = f'{attr}: {raw_val}'
                else:
                    color_val = raw_val
                    hover_text = f'{attr}: {raw_val:.3f}'

                # Create box vertices and faces
                vertices, faces = self._create_box_mesh(x, y, z, dx, dy, dz,
                                                        color_val, plotly_colorscale, vmin, vmax)

                # Add vertices
                all_x.extend(vertices[:, 0])
                all_y.extend(vertices[:, 1])
                all_z.extend(vertices[:, 2])

                # Add faces with offset
                for face in faces:
                    all_i.append(face[0] + vertex_offset)
                    all_j.append(face[1] + vertex_offset)
                    all_k.append(face[2] + vertex_offset)
                    all_colors.append(color_val)

                vertex_offset += 8  # 8 vertices per box

            # Create mesh3d trace
            if attr_type == 'categorical':
                # Categorical: show discrete legend
                trace = go.Mesh3d(
                    x=all_x,
                    y=all_y,
                    z=all_z,
                    i=all_i,
                    j=all_j,
                    k=all_k,
                    intensity=all_colors,
                    colorscale=plotly_colorscale,
                    opacity=opacity,
                    name=attr,
                    colorbar=dict(
                        title=attr,
                        thickness=20,
                        len=0.7,
                        x=1.02,
                        tickmode='array',
                        tickvals=list(range(len(unique_vals))),
                        ticktext=[str(v) for v in unique_vals]
                    ),
                    hovertemplate='<b>%{text}</b><br>' +
                                  'X: %{x:.1f}<br>' +
                                  'Y: %{y:.1f}<br>' +
                                  'Z: %{z:.1f}<br>' +
                                  '<extra></extra>',
                    text=[f'{attr}: {unique_vals[int(c)]}' if c < len(unique_vals) else f'{attr}: N/A'
                          for c in all_colors],
                    visible=(attr_idx == 0)
                )
            else:
                # Numeric: show gradient colorbar
                trace = go.Mesh3d(
                    x=all_x,
                    y=all_y,
                    z=all_z,
                    i=all_i,
                    j=all_j,
                    k=all_k,
                    intensity=all_colors,
                    colorscale=plotly_colorscale,
                    opacity=opacity,
                    name=attr,
                    colorbar=dict(
                        title=attr,
                        thickness=20,
                        len=0.7,
                        x=1.02
                    ),
                    hovertemplate=f'<b>{attr}: %{{intensity:.3f}}</b><br>' +
                                  'X: %{x:.1f}<br>' +
                                  'Y: %{y:.1f}<br>' +
                                  'Z: %{z:.1f}<br>' +
                                  '<extra></extra>',
                    visible=(attr_idx == 0)
                )

            traces.append(trace)

        # Create figure
        fig = go.Figure(data=traces)

        # Create dropdown menu
        buttons = []
        for i, attr in enumerate(attributes):
            if attr not in df_viz.columns or not pd.api.types.is_numeric_dtype(df_viz[attr]):
                continue

            # Create visibility list
            visible = [False] * len(traces)
            visible[i] = True

            buttons.append(
                dict(
                    label=attr,
                    method='update',
                    args=[
                        {'visible': visible},
                        {'title': f'3D Block Model - {attr}'}
                    ]
                )
            )

        # Update layout with dropdown menu
        if title is None:
            title = f'3D Block Model - {attributes[0] if attributes else "Blocks"}'

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title=self.coord_cols['x'],
                yaxis_title=self.coord_cols['y'],
                zaxis_title=self.coord_cols['z'],
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction='down',
                    showactive=True,
                    x=0.15,
                    xanchor='left',
                    y=1.15,
                    yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='rgba(0, 0, 0, 0.2)',
                    borderwidth=1,
                    font=dict(size=11)
                )
            ],
            width=1400,
            height=900,
            hovermode='closest',
            margin=dict(l=0, r=200, t=100, b=0)
        )

        # Add annotation for dropdown menu
        fig.add_annotation(
            text="Select Attribute:",
            x=0.01,
            y=1.15,
            xref='paper',
            yref='paper',
            showarrow=False,
            font=dict(size=12, color='black'),
            xanchor='left',
            yanchor='top'
        )

        return fig

    def visualize_scatter(self, color_by='ni', size_by=None, marker_size=3,
                         colorscale='Viridis', title=None, opacity=0.8,
                         show_colorbar=True):
        """
        Create fast 3D point cloud visualization

        Args:
            color_by (str): Column name to use for coloring
            size_by (str): Column name to use for marker size (optional)
            marker_size (float): Base marker size
            colorscale (str): Plotly colorscale name
            title (str): Plot title
            opacity (float): Marker opacity (0-1)
            show_colorbar (bool): Show colorbar

        Returns:
            plotly.graph_objects.Figure
        """
        if color_by not in self.df.columns:
            print(f"Warning: Column '{color_by}' not found. Available columns:")
            print(f"Numeric: {self.get_numeric_columns()}")
            print(f"Categorical: {self.get_categorical_columns()}")
            raise ValueError(f"Column '{color_by}' not found in data")

        x = self.df[self.coord_cols['x']]
        y = self.df[self.coord_cols['y']]
        z = self.df[self.coord_cols['z']]
        color_values = self.df[color_by]

        # Create hover text with multiple attributes
        hover_cols = ['zone', 'ni', 'co', 'fe', 'products', 'block_value']
        hover_text = []

        for idx, row in self.df.iterrows():
            text_parts = [f"<b>Block Info</b><br>"]
            text_parts.append(f"X: {row[self.coord_cols['x']]:.2f}<br>")
            text_parts.append(f"Y: {row[self.coord_cols['y']]:.2f}<br>")
            text_parts.append(f"Z: {row[self.coord_cols['z']]:.2f}<br>")
            text_parts.append(f"<b>{color_by}: {row[color_by]}</b><br>")

            for col in hover_cols:
                if col in self.df.columns and col != color_by:
                    text_parts.append(f"{col}: {row[col]}<br>")

            hover_text.append("".join(text_parts))

        # Determine if color_by is numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(self.df[color_by])

        if is_numeric:
            # Numeric coloring
            marker = dict(
                size=marker_size,
                color=color_values,
                colorscale=colorscale,
                opacity=opacity,
                showscale=show_colorbar,
                colorbar=dict(title=color_by, thickness=20, len=0.7)
            )
        else:
            # Categorical coloring
            unique_values = self.df[color_by].unique()
            colors = px.colors.qualitative.Plotly
            color_map = {val: colors[i % len(colors)] for i, val in enumerate(unique_values)}
            marker_colors = [color_map[val] for val in color_values]

            marker = dict(
                size=marker_size,
                color=marker_colors,
                opacity=opacity,
                showscale=False
            )

        # Adjust size if size_by is specified
        if size_by and size_by in self.df.columns:
            size_values = self.df[size_by]
            # Normalize size values
            size_normalized = ((size_values - size_values.min()) /
                             (size_values.max() - size_values.min()) * 10 + 2)
            marker['size'] = size_normalized

        # Create 3D point cloud
        scatter = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=marker,
            text=hover_text,
            hoverinfo='text',
            name='Blocks'
        )

        # Create figure
        fig = go.Figure(data=[scatter])

        # Update layout
        if title is None:
            title = f"3D Block Model - Colored by {color_by}"

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            scene=dict(
                xaxis_title=self.coord_cols['x'],
                yaxis_title=self.coord_cols['y'],
                zaxis_title=self.coord_cols['z'],
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1200,
            height=800,
            hovermode='closest'
        )

        return fig

    def get_statistics(self, columns=None):
        """
        Get basic statistics for specified columns

        Args:
            columns (list): List of column names, None for all numeric columns

        Returns:
            pandas.DataFrame: Statistics summary
        """
        if columns is None:
            columns = self.get_numeric_columns()

        stats = self.df[columns].describe()
        return stats

    def save_html(self, fig, output_file='block_model_boxes.html'):
        """Save visualization to HTML"""
        output_path = Path(output_file)
        fig.write_html(str(output_path))
        print(f"Visualization saved to: {output_path.absolute()}")

    def show(self, fig):
        """Display visualization in browser"""
        fig.show()


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description='Advanced 3D Block Model Visualizer with Interactive Menus'
    )

    parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    parser.add_argument('-a', '--attributes', nargs='+',
                       default=['ni', 'co', 'fe', 'mgo', 'sio2', 'block_value'],
                       help='Attributes to include in dropdown menu')
    parser.add_argument('-s', '--sample', type=int, help='Sample size (optional)')
    parser.add_argument('-m', '--max-blocks', type=int, default=None,
                       help='Max blocks to render (default: None = all blocks)')
    parser.add_argument('-o', '--output', default='block_model_boxes.html',
                       help='Output HTML file')
    parser.add_argument('--opacity', type=float, default=0.9,
                       help='Box opacity (default: 0.9)')
    parser.add_argument('--colorscale', default='Viridis',
                       help='Colorscale (default: Viridis)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not open browser')
    parser.add_argument('--skip-rows', type=int, default=3,
                       help='Rows to skip (default: 3)')

    args = parser.parse_args()

    try:
        # Initialize
        viz = BlockModelVisualizer(args.input, skip_rows=args.skip_rows)
        viz.load_data()

        # Sample if requested
        if args.sample:
            viz.sample_data(n_samples=args.sample)

        # Create visualization
        print(f"\nCreating 3D box visualization...")
        fig = viz.create_box_visualization(
            attributes=args.attributes,
            colorscale=args.colorscale,
            opacity=args.opacity,
            max_blocks=args.max_blocks
        )

        # Save
        viz.save_html(fig, args.output)

        # Show
        if not args.no_show:
            print("Opening in browser...")
            viz.show(fig)

        print("\nDone! Use the dropdown menu to change attributes.")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
