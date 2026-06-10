"""
PRISM Frontend — Streamlit Dashboard (Phase 1)

Launch with:
    streamlit run frontend/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="PRISM — Pediatric Respiratory Intelligence System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    """Main dashboard entrypoint."""
    st.title("🫁 PRISM Dashboard")
    st.subheader("Pediatric Respiratory Intelligence System")

    st.info(
        "Welcome to PRISM. This dashboard is under active development. "
        "Use the sidebar to navigate between sections.",
        icon="🔬",
    )

    # ---- Overview Metrics ----
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Recordings", "—", help="Total audio recordings ingested")
    with col2:
        st.metric("Events Detected", "—", help="Total cough events detected")
    with col3:
        st.metric("Subjects", "—", help="Total subjects across all datasets")
    with col4:
        st.metric("Datasets", "3", help="COUGHVID, Coswara, ICBHI")

    st.divider()

    st.markdown(
        """
        ### 📊 System Status

        | Module | Status |
        |--------|--------|
        | Audio Processing Engine | 🔴 Not Started |
        | Cough Detection (CNN) | 🔴 Not Started |
        | Temporal Intelligence | 🔴 Not Started |
        | Environmental Correlation | 🔴 Not Started |
        | Retrieval Engine (RATM) | 🔴 Not Started |
        | Dashboard | 🟡 In Progress |
        """
    )


if __name__ == "__main__":
    main()
