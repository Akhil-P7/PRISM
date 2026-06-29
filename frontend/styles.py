"""
PRISM Frontend — CSS Theme Injection

Provides inject_css() and inject_sidebar_nav() used by all pages.
"""

import streamlit as st


def inject_css():
    """Inject the PRISM clinical UI theme via st.markdown."""
    st.markdown(
        """
        <style>
        /* Import Inter and JetBrains Mono */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        /* Color Tokens */
        :root {
            --prism-bg: #0d1b2a;
            --prism-card: #142232;
            --prism-hover: #1a2e44;
            --prism-border: #1e3a52;
            --prism-accent: #00b4d8;
            --prism-accent-dim: rgba(0, 180, 216, 0.12);
            --prism-text: #e8f4fd;
            --prism-text-muted: #8bb8d4;
            --prism-text-dim: #4a7a9b;

            --severity-info: #38bdf8;
            --severity-low: #22c55e;
            --severity-moderate: #f59e0b;
            --severity-high: #ef4444;
        }

        /* Base Typography */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: var(--prism-text);
            overflow-y: scroll !important;
        }

        /* Streamlit Overrides */
        .stApp {
            background-color: var(--prism-bg) !important;
        }

        .stSidebar {
            background-color: var(--prism-card) !important;
            border-right: 1px solid var(--prism-border) !important;
        }

        /* Metric Cards */
        [data-testid="stMetric"] {
            background-color: var(--prism-card);
            border: 1px solid var(--prism-border);
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }

        /* Buttons */
        .stButton>button {
            background-color: var(--prism-card);
            border: 1px solid var(--prism-accent);
            color: var(--prism-accent);
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.2s ease;
        }

        .stButton>button:hover {
            background-color: var(--prism-accent-dim);
            color: var(--prism-accent);
            border-color: var(--prism-accent);
        }

        /* Custom PRISM Classes */
        .prism-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
            color: var(--prism-text);
        }

        .prism-subtitle {
            font-size: 1.1rem;
            color: var(--prism-accent);
            margin-bottom: 2rem;
            font-weight: 500;
        }

        .prism-card {
            background-color: var(--prism-card);
            border: 1px solid var(--prism-border);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        .prism-report-header {
            border-bottom: 1px solid var(--prism-border);
            padding-bottom: 1rem;
            margin-bottom: 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .prism-report-title {
            font-weight: 600;
            font-size: 1.2rem;
            color: var(--prism-text);
            margin: 0;
        }

        .prism-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-info { background-color: rgba(56, 189, 248, 0.15); color: var(--severity-info); border: 1px solid rgba(56, 189, 248, 0.3); }
        .badge-low { background-color: rgba(34, 197, 94, 0.15); color: var(--severity-low); border: 1px solid rgba(34, 197, 94, 0.3); }
        .badge-moderate { background-color: rgba(245, 158, 11, 0.15); color: var(--severity-moderate); border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-high { background-color: rgba(239, 68, 68, 0.15); color: var(--severity-high); border: 1px solid rgba(239, 68, 68, 0.3); }

        .prism-obs-row {
            display: flex;
            align-items: flex-start;
            padding: 1rem 0;
            border-bottom: 1px solid rgba(30, 58, 82, 0.5);
        }

        .prism-obs-row:last-child {
            border-bottom: none;
        }

        .prism-obs-badge {
            flex: 0 0 100px;
        }

        .prism-obs-text {
            flex: 1;
            padding-left: 1.5rem;
            line-height: 1.5;
        }

        .prism-prob-row {
            display: flex;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .prism-prob-label {
            flex: 0 0 150px;
            font-weight: 500;
        }

        .prism-prob-bar-container {
            flex: 1;
            height: 12px;
            background-color: var(--prism-bg);
            border-radius: 6px;
            overflow: hidden;
            margin: 0 1rem;
            border: 1px solid var(--prism-border);
        }

        .prism-prob-bar {
            height: 100%;
            background-color: var(--prism-accent);
            transition: width 0.5s ease-out;
        }

        .prism-prob-val {
            flex: 0 0 50px;
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: var(--prism-text-muted);
        }

        /* Sidebar Elements */
        .sidebar-logo {
            font-size: 2rem;
            font-weight: 700;
            color: var(--prism-text);
            margin-bottom: 0;
            line-height: 1;
        }
        .sidebar-version {
            font-size: 0.8rem;
            color: var(--prism-text-dim);
            margin-bottom: 2rem;
            font-family: 'JetBrains Mono', monospace;
        }

        /* Clean up some native Streamlit margins */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
            min-height: 101vh !important; /* Force scrollbar to prevent UI shaking */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_sidebar_nav(current_page: str = "home"):
    """Inject custom sidebar navigation."""
    st.sidebar.markdown(
        """
        <div class="sidebar-logo">🫁 PRISM</div>
        <div class="sidebar-version">v0.1.0-alpha</div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.page_link("app.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/insights.py", label="Clinical Insights", icon="🔬")

    st.sidebar.divider()

    st.sidebar.markdown(
        """
        <div style="font-size: 0.75rem; color: var(--prism-text-dim); font-weight: 600;
             text-transform: uppercase; margin-bottom: 0.5rem; letter-spacing: 0.05em;">
            Model Status
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.success("CNN Detector  ✓", icon="🟢")
    st.sidebar.success("RATM Engine   ✓", icon="🟢")
    st.sidebar.success("Classifier    ✓", icon="🟢")
