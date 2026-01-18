# Block Model Column Calculator

Interactive web-based tool for mining block model visualization using 3D point clouds. Supports thickness and stripping ratio calculations for pit optimization.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

- 🎯 **3D Point Cloud Visualization** - Fast rendering for large datasets
- 📊 **Block Sum (Vertical)** - Collapse vertical blocks per (X,Y) column
- 📏 **Thickness Calculation** - Compute thickness per material category
- ⚖️ **Stripping Ratio Calculation** - Compute SR (OB/Ore) per column for pit optimization
- 🎨 **Interactive 3D Controls** - Rotate, zoom, and pan with the mouse
- 📁 **Multi-format Support** - Surpac, Datamine, and standard CSV
- 💾 **Export Results** - Export calculations to CSV with metadata
- 🎨 **Modern UI** - Clean, professional interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or later
- pip (Python package manager)

### Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/blockmodel.git
cd blockmodel
```

2. Install dependencies:
```bash
pip install streamlit pandas plotly numpy
```

### Running the Application

```bash
streamlit run app_streamlit.py
```

The app opens in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1️⃣ Upload Block Model
- Upload a CSV file (Surpac/Datamine format)
- Set "Rows to skip" for metadata (default: 3)

### 2️⃣ Block Sum Configuration
- **Sum All**: Standard summation of all vertical blocks
- **Calculate Thickness**: Calculate thickness per category (OB, Ore, Waste)
- **Calculate Stripping Ratio**: Calculate SR = thickness_OB / thickness_Ore per column

### 3️⃣ Visualize
- Select an attribute for visualization (grade, tonnage, SR, etc.)
- 3D point cloud appears automatically
- Use the mouse to rotate, zoom, and pan

### 4️⃣ Export
- Download calculation results as CSV
- Includes new columns: `thickness_OB`, `thickness_Ore`, `stripping_ratio`

## ⚠️ Important Notes

1. **Block Models only**; Stratigraphic Models are not supported
2. **Format**: only .csv (Surpac, Datamine, or standard format)
3. **Visualization**: uses Point Cloud for optimal performance
4. **Thickness/SR**: only supports categorical attributes (Text/String)

## 📁 File Structure

```
blockmodel/
├── app_streamlit.py              # Main Streamlit web app
├── block_model_visualizer.py     # Core visualization engine
├── README.md                      # Documentation
├── .gitignore                     # Git ignore rules
└── example_data/                  # Example CSV files (optional)
```

## 🛠️ Technical Details

### Supported CSV Formats
- Datamine (with metadata rows)
- Vulcan
- Surpac
- Standard CSV with X, Y, Z coordinates

### Required Columns
- Coordinate columns: `xc`, `yc`, `zc` (or `centroid_x`, `centroid_y`, `centroid_z`)
- Dimension column (optional): `dz`, `zinc`, `dim_z` for thickness calculation
- Categorical attributes (optional): for thickness/SR calculation

### Performance
- Optimal: < 100,000 blocks
- Good: 100,000 - 500,000 blocks
- Slow: > 500,000 blocks (use Block Sum for optimization)

## 🎨 Screenshots

*(Add screenshots here if available)*

## 📝 License

MIT License - feel free to use and modify for your projects

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 🐛 Bug Reports

Found a bug? Please open an issue on GitHub with:
- Steps to reproduce
- Expected behavior
- Actual behavior
- CSV file format (if relevant)

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with** ❤️ **using Streamlit, Plotly, and Python**
