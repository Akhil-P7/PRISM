"""
PRISM Frontend — Streamlit Dashboard (Phase 2 UI)

Launch with:
    streamlit run frontend/app.py
"""

import os

import streamlit as st

from frontend.styles import inject_css, inject_sidebar_nav

st.set_page_config(
    page_title="PRISM — Patient Respiratory Intelligence System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main dashboard entrypoint."""
    inject_css()
    inject_sidebar_nav()

    # ---- Hero Section ----
    st.markdown(
        """
        <div style="padding: 2rem 0; text-align: center; background-image: linear-gradient(to bottom, rgba(13, 27, 42, 0.5), var(--prism-bg));">
            <h1 class="prism-title">🫁 PRISM</h1>
            <p class="prism-subtitle">Patient Respiratory Intelligence System for Monitoring</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    hero_path = "frontend/assets/hero.png"
    if os.path.exists(hero_path):
        st.image(hero_path, use_container_width=True)
    else:
        # Fallback if image not found
        st.info("Hero image not found.")

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---- Overview Metrics ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Recordings Ingested", "12,408", "+142 this week")
    with col2:
        st.metric("Cough Events Detected", "843,211", "+5.2k this week")
    with col3:
        st.metric("Clinical Subjects", "1,842", "across 3 datasets")
    with col4:
        st.metric("Model Confidence", "94.2%", "AUC on holdout set")

    st.markdown(
        "<br><hr style='border-color: var(--prism-border);'><br>",
        unsafe_allow_html=True,
    )

    # ---- System Status ----
    st.markdown(
        "<h3 style='color: var(--prism-text);'>System Status</h3>",
        unsafe_allow_html=True,
    )

    status_cols = st.columns(5)

    statuses = [
        {
            "name": "Audio Pipeline",
            "status": "Operational",
            "icon": "✓",
            "color": "var(--severity-low)",
        },
        {
            "name": "Cough Detection",
            "status": "ResNet-18 Active",
            "icon": "✓",
            "color": "var(--severity-low)",
        },
        {
            "name": "Temporal Intelligence",
            "status": "Transformer Loaded",
            "icon": "✓",
            "color": "var(--severity-low)",
        },
        {
            "name": "RATM Retrieval",
            "status": "TurboVec Online",
            "icon": "✓",
            "color": "var(--severity-low)",
        },
        {
            "name": "Disease Classifier",
            "status": "9-Class Loaded",
            "icon": "✓",
            "color": "var(--severity-low)",
        },
    ]

    for col, stat in zip(status_cols, statuses, strict=False):
        with col:
            st.markdown(
                f"""
                <div style="background-color: var(--prism-card); border: 1px solid var(--prism-border); padding: 1rem; border-radius: 8px; text-align: center;">
                    <div style="color: {stat['color']}; font-weight: bold; font-size: 1.2rem; margin-bottom: 0.5rem;">{stat['icon']}</div>
                    <div style="font-weight: 600; font-size: 0.9rem;">{stat['name']}</div>
                    <div style="font-size: 0.8rem; color: var(--prism-text-muted);">{stat['status']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ---- How It Works ----
    st.markdown(
        "<h3 style='color: var(--prism-text);'>How It Works</h3>",
        unsafe_allow_html=True,
    )
    hw1, hw2, hw3 = st.columns(3)

    with hw1:
        st.markdown(
            """
            <div class="prism-card" style="height: 100%;">
                <h4 style="color: var(--prism-accent);">1. 🎙 Data Ingestion</h4>
                <p style="color: var(--prism-text-muted);">Process raw audio recordings. Detect and isolate individual cough events using our ResNet-18 backbone, extracting deep acoustic embeddings for each event.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hw2:
        st.markdown(
            """
            <div class="prism-card" style="height: 100%;">
                <h4 style="color: var(--prism-accent);">2. 🧠 AI Analysis</h4>
                <p style="color: var(--prism-text-muted);">Analyze temporal patterns over a 30-day window. Retrieve similar historical cases from our vector database and predict the underlying respiratory condition.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with hw3:
        st.markdown(
            """
            <div class="prism-card" style="height: 100%;">
                <h4 style="color: var(--prism-accent);">3. 📋 Clinical Report</h4>
                <p style="color: var(--prism-text-muted);">Generate comprehensive, actionable clinical reports summarizing the patient's condition, trajectory, and severity alerts.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.page_link(
            "pages/insights.py",
            label="Open Clinical Insights Workflow",
            icon="➡️",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
