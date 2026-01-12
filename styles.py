"""
CSS Styles for Block Model Visualizer Streamlit App

This module contains all CSS styling for the application.
"""


def get_custom_css() -> str:
    """
    Returns the complete CSS stylesheet for the Streamlit app

    Returns:
        str: CSS as a string to be injected via st.markdown()
    """
    return """
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

    /* Hide sidebar collapse button to keep sidebar always visible */
    button[kind="headerNoPadding"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}
</style>
"""
