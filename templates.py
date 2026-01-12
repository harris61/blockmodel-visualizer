"""
HTML Templates for Block Model Visualizer Streamlit App

This module contains all HTML template generators for the application.
"""


def get_header_html() -> str:
    """
    Returns the main header/title HTML

    Returns:
        str: HTML string for the header section
    """
    return """
<div style='text-align: center; padding: 1.5rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <h1 style='color: white; margin: 0; font-size: 2.5rem; font-weight: 700;'>📦 3D Block Model Visualizer</h1>
    <p style='color: #e0e7ff; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>Interactive web interface untuk visualisasi block model tambang</p>
</div>
"""


def get_3d_controls_html() -> str:
    """
    Returns the 3D controls guide HTML

    Returns:
        str: HTML string for 3D controls guide
    """
    return """
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
"""


def get_important_notes_html() -> str:
    """
    Returns the important notes section HTML

    Returns:
        str: HTML string for important notes
    """
    return """
<div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 6px solid #dc2626; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
    <h2 style='color: #991b1b; margin-top: 0;'>⚠️ Catatan Penting</h2>
    <div style='background: white; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
        <ol style='color: #374151; margin: 0; line-height: 1.8;'>
            <li><strong>Hanya bisa mengolah Blok Model</strong>, tidak bisa mengolah Stratigraphic Model</li>
            <li><strong>Format file yang didukung:</strong> Blok Model hanya dalam format .csv, baik dalam format Surpac, Datamine, dll</li>
            <li><strong>Visualisasi menggunakan Point Cloud</strong> untuk alasan performa</li>
            <li><strong>Tipe Data Attribute Klasifikasi Material:</strong> attribute untuk klasifikasi material hanya bisa membaca attribute dengan tipe data teks</li>
        </ol>
        <div style='background: #fef3c7; padding: 0.75rem; border-radius: 6px; margin-top: 1rem; border-left: 4px solid #f59e0b;'>
            <p style='color: #78350f; margin: 0;'><strong>💡 Tips:</strong> Buat attribute di dalam block model berupa klasifikasi material dengan tipe data teks terlebih dahulu</p>
        </div>
    </div>
</div>
"""


def get_user_guide_header_html() -> str:
    """
    Returns the user guide header HTML

    Returns:
        str: HTML string for user guide header
    """
    return """
<div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 6px solid #f59e0b;'>
    <h2 style='color: #92400e; margin-top: 0;'>📖 Panduan Penggunaan</h2>
    <p style='color: #78350f; margin-bottom: 0;'>Ikuti langkah-langkah berikut untuk memulai visualisasi block model Anda</p>
</div>
"""


def get_step1_html() -> str:
    """Step 1: Upload & Setup"""
    return """
<div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem; border-left: 5px solid #667eea;'>
    <h3 style='color: #667eea; margin-top: 0;'>1️⃣ Upload & Setup</h3>
    <ul style='color: #4b5563;'>
        <li>Upload file CSV (Surpac/Datamine/format standar)</li>
        <li>Atur "Number of Header Lines"</li>
        <li>Pilih <strong>Colorscale</strong> dan <strong>Color Mode</strong> (Auto/Gradient/Discrete)</li>
    </ul>
</div>
"""


def get_step2_html() -> str:
    """Step 2: Block Sum Configuration"""
    return """
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
"""


def get_step3_html() -> str:
    """Step 3: Visualisasi Point Cloud"""
    return """
<div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem; border-left: 5px solid #f59e0b;'>
    <h3 style='color: #f59e0b; margin-top: 0;'>3️⃣ Visualisasi Point Cloud</h3>
    <ul style='color: #4b5563;'>
        <li>Pilih attribute dari <strong>Numeric</strong> (grade, tonnage, SR) atau <strong>Categorical</strong> (lithology, zone)</li>
        <li>Visualisasi 3D otomatis muncul</li>
        <li>Kontrol: <strong>Drag</strong> (rotasi) | <strong>Scroll</strong> (zoom) | <strong>Right-drag</strong> (pan) | <strong>Double-click</strong> (reset)</li>
    </ul>
</div>
"""


def get_step4_html() -> str:
    """Step 4: Filter & Export"""
    return """
<div style='background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 1.5rem; border-left: 5px solid #8b5cf6;'>
    <h3 style='color: #8b5cf6; margin-top: 0;'>4️⃣ Filter & Export</h3>
    <ul style='color: #4b5563;'>
        <li>Filter data berdasarkan range attribute</li>
        <li>Export hasil dengan <strong>"Export CSV"</strong> (termasuk kolom thickness/SR)</li>
    </ul>
</div>
"""


def get_footer_html(logo_base64: str) -> str:
    """
    Returns the footer HTML

    Args:
        logo_base64 (str): Base64 encoded logo image

    Returns:
        str: HTML string for footer
    """
    return f"""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 8px; text-align: center;'>
    <p style='color: white; font-weight: 700; font-size: 1.1rem; margin: 0;'>Block Model Visualizer</p>
    <p style='color: #e0e7ff; font-size: 0.9rem; margin: 0.5rem 0 1rem 0;'>v1.0</p>
    <div style='border-top: 1px solid rgba(255,255,255,0.3); padding-top: 1rem; padding-bottom: 1rem;'>
        <p style='color: #e0e7ff; font-size: 0.8rem; margin: 0.3rem 0 0.2rem 0;'>Created by</p>
        <a href='https://www.linkedin.com/in/harristio-adam/' target='_blank' style='color: white; text-decoration: none; font-weight: 600; font-size: 0.85rem;'>Harristio Adam</a>
        <p style='color: #e0e7ff; font-size: 0.8rem; margin: 0.8rem 0 0.3rem 0;'>Powered by</p>
        <a href='https://www.linkedin.com/company/soft-roc' target='_blank'>
            <img src='data:image/png;base64,{logo_base64}' style='max-width: 150px; height: auto;' alt='Soft.Roc'>
        </a>
    </div>
</div>
"""
