"""
HTML Templates for Block Model Column Calculator Streamlit App

This module contains all HTML template generators for the application.
"""


def get_header_html(app_version: str) -> str:
    """
    Returns the main header/title HTML

    Returns:
        str: HTML string for the header section
    """
    return f"""
<div style='text-align: left; padding: 0.5rem 1rem; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; margin-bottom: 0.75rem;'>
    <h1 style='color: #0f172a; margin: 0; font-size: 2rem; font-weight: 700;'>Block Model Column Calculator <span style='color: #64748b; font-weight: 600; font-size: 1rem;'>{app_version}</span></h1>
    <p style='color: #475569; margin: 0.25rem 0 0 0; font-size: 1rem;'>Interactive web interface for block model visualization</p>
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
    <h2 style='color: #0f172a; margin-top: 0;'>Important Notes</h2>
    <div style='background: #f8fafc; padding: 1rem; border-radius: 8px; margin-top: 0.75rem;'>
        <ol style='color: #334155; margin: 0; line-height: 1.8;'>
            <li><strong>Only supports Block Models</strong>: Stratigraphic Models are not supported</li>
            <li><strong>Supported file format:</strong> Block Models must be provided as .csv (Surpac, Datamine, etc.)</li>
            <li><strong>Visualization uses point clouds</strong> for performance</li>
            <li><strong>Material classification attributes:</strong> classification attributes must be text/string type</li>
        </ol>
        <div style='background: #fff7ed; padding: 0.75rem; border-radius: 6px; margin-top: 1rem; border-left: 3px solid #f59e0b;'>
            <p style='color: #7c2d12; margin: 0;'><strong>Tip:</strong> Create a text-based material classification attribute in the block model first</p>
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
    <h2 style='color: #0f172a; margin-top: 0;'>User Guide</h2>
    <p style='color: #475569; margin-bottom: 0;'>Follow these steps to start visualizing your block model</p>
</div>
"""


def get_step1_html() -> str:
    """Step 1: Upload & Setup"""
    return """
<div style='background: #ffffff; padding: 1.25rem; border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 1rem; border: 1px solid #e2e8f0;'>
    <h3 style='color: #1e293b; margin-top: 0;'>1. Upload & Setup</h3>
    <ul style='color: #475569;'>
        <li>Upload a CSV file (Surpac/Datamine/standard format)</li>
        <li>Set "Number of Header Lines"</li>
        <li>Choose <strong>Colorscale</strong> and <strong>Color Mode</strong> (Auto/Gradient/Discrete)</li>
    </ul>
</div>
"""


def get_step2_html() -> str:
    """Step 2: Block Sum Configuration"""
    return """
<div style='background: #ffffff; padding: 1.25rem; border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 1rem; border: 1px solid #e2e8f0;'>
    <h3 style='color: #1e293b; margin-top: 0;'>2. Block Sum Configuration</h3>
    <p style='color: #475569; font-weight: 600; margin-top: 1rem;'>Calculate Thickness mode:</p>
    <ul style='color: #475569; margin-top: 0.5rem;'>
        <li>Select a categorical attribute (lithology/zone)</li>
        <li>Select categories to calculate thickness</li>
        <li>Result: new column <code>thickness_{category}</code></li>
    </ul>
    <p style='color: #475569; font-weight: 600; margin-top: 1rem;'>Calculate Stripping Ratio mode:</p>
    <ul style='color: #475569; margin-top: 0.5rem;'>
        <li>Select OB/Waste and Ore categories</li>
        <li>Result: <code>thickness_OB</code>, <code>thickness_Ore</code>, <code>stripping_ratio</code></li>
        <li>SR = thickness_OB / thickness_Ore per column</li>
    </ul>
    <p style='color: #475569; font-weight: 600; margin-top: 1rem;'>Calculate Block Sum mode:</p>
    <ul style='color: #475569; margin-top: 0.5rem;'>
        <li>Select a categorical attribute (e.g., material_type)</li>
        <li>Select categories to include (e.g., ore, waste)</li>
        <li>Select a value attribute to sum (e.g., grade, tonnage)</li>
        <li>Result: new column <code>sum_{category}_{attribute}</code> for each category</li>
    </ul>
    <p style='color: #475569; font-weight: 600; margin-top: 1rem;'>Calculate Block Average mode:</p>
    <ul style='color: #475569; margin-top: 0.5rem;'>
        <li>Select a categorical attribute (e.g., material_type)</li>
        <li>Select categories to include (e.g., ore, waste)</li>
        <li>Select a value attribute to average (e.g., grade, tonnage)</li>
        <li>Result: new column <code>avg_{category}_{attribute}</code> for each category</li>
    </ul>
    <p style='color: #2563eb; font-weight: 600; margin-top: 1rem;'>Click "Calculate Block!" and use Reset/Export as needed.</p>
</div>
"""


def get_step3_html() -> str:
    """Step 3: Point Cloud Visualization"""
    return """
<div style='background: #ffffff; padding: 1.25rem; border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 1rem; border: 1px solid #e2e8f0;'>
    <h3 style='color: #1e293b; margin-top: 0;'>3. Point Cloud Visualization</h3>
    <ul style='color: #475569;'>
        <li>Select an attribute from <strong>Numeric</strong> (grade, tonnage, SR) or <strong>Categorical</strong> (lithology, zone)</li>
        <li>3D visualization appears automatically</li>
        <li>Controls: <strong>Drag</strong> (rotate) | <strong>Scroll</strong> (zoom) | <strong>Right-drag</strong> (pan) | <strong>Double-click</strong> (reset)</li>
    </ul>
</div>
"""


def get_step4_html() -> str:
    """Step 4: Filter and Export"""
    return """
<div style='background: #ffffff; padding: 1.25rem; border-radius: 10px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); margin-bottom: 1rem; border: 1px solid #e2e8f0;'>
    <h3 style='color: #1e293b; margin-top: 0;'>4. Filter and Export</h3>
    <ul style='color: #475569;'>
        <li>Filter data by attribute range</li>
        <li>Export results with <strong>"Export CSV"</strong> (including thickness/SR columns)</li>
    </ul>
</div>
"""


def get_footer_html(logo_base64: str, app_version: str) -> str:
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
    <p style='color: #475569; font-size: 0.9rem; margin: 0.4rem 0 0.8rem 0;'>{app_version}</p>
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
