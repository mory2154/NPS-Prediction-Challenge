"""
Streamlit UI for the NPS prediction system.

This is a placeholder for Phase 12. Run with:
    streamlit run app/main.py
"""

import streamlit as st

st.set_page_config(
    page_title="NPS Prediction — Telco",
    page_icon="📊",
    layout="wide",
)

st.title("NPS Prediction — Telco Retention")
st.caption("Phase 0 placeholder · UI built in Phase 12")

st.info(
    "The full interface — customer lookup, prediction, top drivers, "
    "verbatim impact — will be implemented in Phase 12 once the model is trained."
)

st.subheader("Setup verification")
try:
    from src.config import (
        RANDOM_SEED, NPS_CLASSES, LEAKY_FEATURES, DATA_RAW,
    )
    st.success("✓ `src.config` imports correctly.")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Random seed", RANDOM_SEED)
        st.metric("NPS classes", len(NPS_CLASSES))
    with col2:
        st.metric("Leaky features dropped", len(LEAKY_FEATURES))
        st.metric("Raw data path exists", str(DATA_RAW.exists()))
    with st.expander("Leaky features (auto-dropped before modelling)"):
        st.write(LEAKY_FEATURES)
except ImportError as e:
    st.error(f"✗ Could not import `src.config`: {e}")
    st.write("Run from project root: `streamlit run app/main.py`")
