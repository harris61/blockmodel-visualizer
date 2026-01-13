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
<div style='text-align: left; padding: 0.5rem 1rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 0.75rem;'>
    <h1 style='color: #0f172a; margin: 0; font-size: 2rem; font-weight: 700;'>Block Model Column Calculator</h1>
    <p style='color: #475569; margin: 0.25rem 0 0 0; font-size: 1rem;'>Interactive web interface untuk visualisasi block model tambang</p>
</div>
"""


def get_3d_controls_html() -> str:
    """
    Returns the 3D controls guide HTML

    Returns:
        str: HTML string for 3D controls guide
    """
    return """
<div style='background: #f8fafc; padding: 1rem; border-radius: 8px; border: 1px solid #e2e8f0; margin-top: 1rem;'>
    <h4 style='color: #0f172a; margin-top: 0;'>3D Controls</h4>
    <ul style='color: #1f2937; margin-bottom: 0;'>
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
<div style='background: #ffffff; padding: 1.25rem; border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #e2e8f0;'>
    <h2 style='color: #0f172a; margin-top: 0;'>Catatan Penting</h2>
    <div style='background: #f8fafc; padding: 1rem; border-radius: 8px; margin-top: 0.75rem;'>
        <ol style='color: #334155; margin: 0; line-height: 1.8;'>
            <li><strong>Hanya bisa mengolah Blok Model</strong>, tidak bisa mengolah Stratigraphic Model</li>
            <li><strong>Format file yang didukung:</strong> Blok Model hanya dalam format .csv, baik dalam format Surpac, Datamine, dll</li>
            <li><strong>Visualisasi menggunakan Point Cloud</strong> untuk alasan performa</li>
            <li><strong>Tipe Data Attribute Klasifikasi Material:</strong> attribute untuk klasifikasi material hanya bisa membaca attribute dengan tipe data teks</li>
        </ol>
        <div style='background: #fff7ed; padding: 0.75rem; border-radius: 6px; margin-top: 1rem; border-left: 3px solid #f59e0b;'>
            <p style='color: #7c2d12; margin: 0;'><strong>Tips:</strong> Buat attribute di dalam block model berupa klasifikasi material dengan tipe data teks terlebih dahulu</p>
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
<div style='background: #ffffff; padding: 1.25rem; border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid #e2e8f0;'>
    <h2 style='color: #0f172a; margin-top: 0;'>Panduan Penggunaan</h2>
    <p style='color: #475569; margin-bottom: 0;'>Ikuti langkah-langkah berikut untuk memulai visualisasi block model Anda</p>
</div>
"""


def get_step1_html() -> str:
    """Step 1: Upload & Setup"""
    return """
<div style='background: #ffffff; padding: 1.25rem; border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 1rem; border: 1px solid #e2e8f0;'>
    <h3 style='color: #1e293b; margin-top: 0;'>1. Upload & Setup</h3>
    <ul style='color: #475569;'>
        <li>Upload file CSV (Surpac/Datamine/format standar)</li>
        <li>Atur "Number of Header Lines"</li>
        <li>Pilih <strong>Colorscale</strong> dan <strong>Color Mode</strong> (Auto/Gradient/Discrete)</li>
    </ul>
</div>
"""


def get_step2_html() -> str:
    """Step 2: Block Sum Configuration"""
    return """
<div style='background: #ffffff; padding: 1.25rem; border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 1rem; border: 1px solid #e2e8f0;'>
    <h3 style='color: #1e293b; margin-top: 0;'>2. Block Sum Configuration</h3>
    <p style='color: #475569; font-weight: 600; margin-top: 1rem;'>Mode Calculate Thickness:</p>
    <ul style='color: #475569; margin-top: 0.5rem;'>
        <li>Pilih categorical attribute (lithology/zone)</li>
        <li>Pilih kategori yang ingin dihitung thickness-nya</li>
        <li>Hasil: kolom baru <code>thickness_{kategori}</code></li>
    </ul>
    <p style='color: #475569; font-weight: 600; margin-top: 1rem;'>Mode Calculate Stripping Ratio:</p>
    <ul style='color: #475569; margin-top: 0.5rem;'>
        <li>Tentukan kategori OB/Waste dan Ore</li>
        <li>Hasil: <code>thickness_OB</code>, <code>thickness_Ore</code>, <code>stripping_ratio</code></li>
        <li>SR = thickness_OB / thickness_Ore per kolom</li>
    </ul>
    <p style='color: #475569; font-weight: 600; margin-top: 1rem;'>Mode Calculate Block Sum:</p>
    <ul style='color: #475569; margin-top: 0.5rem;'>
        <li>Pilih categorical attribute (misalnya: material_type)</li>
        <li>Pilih kategori yang ingin dihitung (misalnya: ore, waste)</li>
        <li>Pilih value attribute yang ingin dijumlahkan (misalnya: grade, tonnage)</li>
        <li>Hasil: kolom baru <code>sum_{kategori}_{attribute}</code> untuk setiap kategori</li>
    </ul>
    <p style='color: #475569; font-weight: 600; margin-top: 1rem;'>Mode Calculate Block Average:</p>
    <ul style='color: #475569; margin-top: 0.5rem;'>
        <li>Pilih categorical attribute (misalnya: material_type)</li>
        <li>Pilih kategori yang ingin dihitung (misalnya: ore, waste)</li>
        <li>Pilih value attribute yang ingin dirata-rata (misalnya: grade, tonnage)</li>
        <li>Hasil: kolom baru <code>avg_{kategori}_{attribute}</code> untuk setiap kategori</li>
    </ul>
    <p style='color: #2563eb; font-weight: 600; margin-top: 1rem;'>Klik "Calculate Block!" dan gunakan Reset/Export sesuai kebutuhan.</p>
</div>
"""


def get_step3_html() -> str:
    """Step 3: Visualisasi Point Cloud"""
    return """
<div style='background: #ffffff; padding: 1.25rem; border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 1rem; border: 1px solid #e2e8f0;'>
    <h3 style='color: #1e293b; margin-top: 0;'>3. Visualisasi Point Cloud</h3>
    <ul style='color: #475569;'>
        <li>Pilih attribute dari <strong>Numeric</strong> (grade, tonnage, SR) atau <strong>Categorical</strong> (lithology, zone)</li>
        <li>Visualisasi 3D otomatis muncul</li>
        <li>Kontrol: <strong>Drag</strong> (rotasi) | <strong>Scroll</strong> (zoom) | <strong>Right-drag</strong> (pan) | <strong>Double-click</strong> (reset)</li>
    </ul>
</div>
"""


def get_step4_html() -> str:
    """Step 4: Filter & Export"""
    return """
<div style='background: #ffffff; padding: 1.25rem; border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 1rem; border: 1px solid #e2e8f0;'>
    <h3 style='color: #1e293b; margin-top: 0;'>4. Filter & Export</h3>
    <ul style='color: #475569;'>
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
<div style='background: #f8fafc; padding: 1rem; border-radius: 8px; text-align: center; border: 1px solid #e2e8f0;'>
    <p style='color: #0f172a; font-weight: 700; font-size: 1.05rem; margin: 0;'>Block Model Column Calculator</p>
    <p style='color: #475569; font-size: 0.9rem; margin: 0.4rem 0 0.8rem 0;'>v1.0</p>
    <div style='border-top: 1px solid #e2e8f0; padding-top: 0.75rem; padding-bottom: 0.75rem;'>
        <p style='color: #64748b; font-size: 0.8rem; margin: 0.3rem 0 0.2rem 0;'>Created by</p>
        <a href='https://www.linkedin.com/in/harristio-adam/' target='_blank' style='color: #1e293b; text-decoration: none; font-weight: 600; font-size: 0.85rem;'>Harristio Adam</a>
        <p style='color: #64748b; font-size: 0.8rem; margin: 0.6rem 0 0.3rem 0;'>Powered by</p>
        <a href='https://www.linkedin.com/company/soft-roc' target='_blank'>
            <img src='data:image/png;base64,{logo_base64}' style='max-width: 150px; height: auto;' alt='Soft.Roc'>
        </a>
    </div>
</div>
"""
