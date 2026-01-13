"""
CSS Styles for Block Model Column Calculator Streamlit App

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
        padding-top: 0.25rem;
        padding-bottom: 1.25rem;
    }

    /* Header styling */
    h1 {
        color: #0f172a;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    h2 {
        color: #1e293b;
        font-weight: 600;
        margin-top: 0.75rem;
        margin-bottom: 0.75rem;
        border-bottom: 1px solid #cbd5e1;
        padding-bottom: 0.4rem;
    }

    h3 {
        color: #1f2937;
        font-weight: 600;
        margin-top: 0.75rem;
    }

    /* Card-like containers */
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        background-color: #ffffff;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        margin-bottom: 0.75rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 0.25rem;
    }

    /* Button styling - apply same gradient style to ALL buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: #2563eb !important;
        color: #ffffff !important;
        border-radius: 6px;
        font-weight: 600;
        border: 1px solid #1d4ed8 !important;
    }

    /* Additional selector for button styling */
    div[data-testid="stButton"] button,
    div[data-testid="stDownloadButton"] button {
        background: #2563eb !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
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
        background-color: #ecfdf3;
        color: #065f46;
        border-left: 3px solid #10b981;
        border-radius: 6px;
    }

    /* Warning message */
    .stWarning {
        background-color: #fffbeb;
        color: #92400e;
        border-left: 3px solid #f59e0b;
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
