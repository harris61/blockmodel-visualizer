# Block Model Column Calculator

Interactive web-based tool untuk visualisasi block model tambang menggunakan point cloud 3D. Mendukung perhitungan thickness dan stripping ratio untuk pit optimization.

![Version](https://img.shields.io/badge/version-2.1-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ Features

- 🎯 **3D Point Cloud Visualization** - Fast rendering untuk dataset besar
- 📊 **Block Sum (Vertical)** - Collapse blok vertikal per kolom (X,Y)
- 📏 **Thickness Calculation** - Hitung ketebalan per kategori material
- ⚖️ **Stripping Ratio Calculation** - Hitung SR (OB/Ore) per kolom untuk pit optimization
- 🎨 **Interactive 3D Controls** - Rotate, zoom, pan dengan mouse
- 📁 **Multi-format Support** - Surpac, Datamine, dan format CSV standar
- 💾 **Export Results** - Export hasil perhitungan ke CSV dengan metadata
- 🎨 **Modern UI** - Clean, professional interface

## 🚀 Quick Start

### Prerequisites

- Python 3.8 atau lebih baru
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

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📖 Usage Guide

### 1️⃣ Upload Block Model
- Upload file CSV (format Surpac/Datamine)
- Atur "Rows to skip" untuk metadata (default: 3)

### 2️⃣ Block Sum Configuration
- **Sum All**: Penjumlahan standar semua blok vertikal
- **Calculate Thickness**: Hitung thickness per kategori (OB, Ore, Waste)
- **Calculate Stripping Ratio**: Hitung SR = thickness_OB / thickness_Ore per kolom

### 3️⃣ Visualize
- Pilih attribute untuk visualisasi (grade, tonnage, SR, dll)
- Point cloud 3D otomatis muncul
- Gunakan mouse untuk rotate, zoom, dan pan

### 4️⃣ Export
- Download hasil perhitungan sebagai CSV
- Termasuk kolom baru: `thickness_OB`, `thickness_Ore`, `stripping_ratio`

## ⚠️ Important Notes

1. **Hanya untuk Block Model**, tidak mendukung Stratigraphic Model
2. **Format**: hanya .csv (Surpac, Datamine, atau format standar)
3. **Visualisasi**: menggunakan Point Cloud untuk performa optimal
4. **Thickness/SR**: hanya mendukung categorical attributes (Text/String)

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
- Coordinate columns: `xc`, `yc`, `zc` (atau `centroid_x`, `centroid_y`, `centroid_z`)
- Dimension column (optional): `dz`, `zinc`, `dim_z` untuk thickness calculation
- Categorical attributes (optional): untuk thickness/SR calculation

### Performance
- Optimal: < 100,000 blocks
- Good: 100,000 - 500,000 blocks
- Slow: > 500,000 blocks (gunakan Block Sum untuk optimisasi)

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
