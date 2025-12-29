"""
Meridian UI - Results Page (Framework Top)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.layout import inject_global_style, kpi_card, pills_bar, section_title, divider, callout, failure_card
from ui.vega_theme import ci_bar, single_boxplot

st.set_page_config(page_title="Results - Meridian", page_icon="M", layout="wide")
inject_global_style()

st.markdown("# Results")
st.markdown('<div class="ce-subtitle">Run summary with statistical confidence, latency distribution, and failure drill-down.</div>', unsafe_allow_html=True)

# Get runs
try:
    from core.storage.db import get_db
    db = get_db()
    runs = db.get_runs(limit=50)
except Exception as e:
    st.error(f"Error: {e}")
    runs = []

if not runs:
    st.markdown(callout("No runs found. Go to <b>Run Suite</b> to evaluate a model."), unsafe_allow_html=True)
    st.stop()

run_options = {r["run_id"]: f"{r['suite_name']} · {r['model_id']} · {r['started_at'][:10]}" for r in runs}
selected_run = st.selectbox("Select run", list(run_options.keys()), format_func=lambda x: run_options[x])

run = next((r for r in runs if r["run_id"] == selected_run), None)
results = db.get_results(selected_run)

if run and results:
    df = pd.DataFrame(results)
    
    # Stats
    accuracy = run['accuracy'] * 100 if run['accuracy'] else 0
    passed = run['passed_tests']
    failed = run['failed_tests']
    n_tests = passed + failed
    
    ci_lo = run['accuracy_ci_lower'] * 100 if run['accuracy_ci_lower'] else accuracy
    ci_hi = run['accuracy_ci_upper'] * 100 if run['accuracy_ci_upper'] else accuracy
    
    latencies = df['latency_ms'].dropna().values / 1000
    p50 = np.percentile(latencies, 50) if len(latencies) > 0 else 0
    p95 = np.percentile(latencies, 95) if len(latencies) > 0 else 0
    
    # Context pills
    st.markdown(pills_bar([
        f"Suite: {run['suite_name']}",
        f"Model: {run['model_id']}",
        f"n={n_tests} tests",
        f"Bootstrap B=1000",
        f"{run['started_at'][:16]}"
    ]), unsafe_allow_html=True)
    
    st.markdown(divider(), unsafe_allow_html=True)
    
    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(kpi_card(
            f"{accuracy:.0f}%", 
            "Accuracy",
            f"95% CI: [{ci_lo:.0f}%, {ci_hi:.0f}%]"
        ), unsafe_allow_html=True)
    
    with c2:
        st.markdown(kpi_card(f"{p50:.2f}s", "p50 Latency"), unsafe_allow_html=True)
    
    with c3:
        st.markdown(kpi_card(f"{p95:.2f}s", "p95 Latency"), unsafe_allow_html=True)
    
    with c4:
        st.markdown(kpi_card(
            f"{passed}/{n_tests}", 
            "Passed",
            f"{failed} failed" if failed > 0 else None
        ), unsafe_allow_html=True)
    
    # Evidence callout
    st.markdown(callout(
        f"<b>Evidence:</b> Accuracy {accuracy:.0f}% "
        f"(95% CI: [{ci_lo:.0f}%, {ci_hi:.0f}%]) · "
        f"Latency p50={p50:.2f}s, p95={p95:.2f}s · "
        f"Evaluated under quasi-deterministic settings."
    ), unsafe_allow_html=True)
    
    st.markdown(divider(), unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown(section_title("Accuracy with Confidence Interval"), unsafe_allow_html=True)
        st.vega_lite_chart(ci_bar(accuracy, ci_lo, ci_hi, n_tests, ""), use_container_width=True)
    
    with col2:
        st.markdown(section_title("Latency Distribution"), unsafe_allow_html=True)
        if len(latencies) > 0:
            st.vega_lite_chart(single_boxplot(list(latencies), ""), use_container_width=True)
    
    st.markdown(divider(), unsafe_allow_html=True)
    
    # Top failures
    failed_df = df[df['passed'] == 0]
    if len(failed_df) > 0:
        st.markdown(section_title(f"Top Failures ({len(failed_df)})"), unsafe_allow_html=True)
        
        for _, row in failed_df.head(3).iterrows():
            output = row.get('output', '')
            reason = output[:100] + "..." if output and len(output) > 100 else output or "No output"
            st.markdown(failure_card(row['test_id'], reason), unsafe_allow_html=True)
    
    st.markdown(divider(), unsafe_allow_html=True)
    
    # All tests table
    st.markdown(section_title("All Tests"), unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        filter_opt = st.selectbox("Filter", ["All", "Passed", "Failed"], label_visibility="collapsed")
    with col2:
        if st.button("Copy Run ID", type="secondary"):
            st.code(selected_run)
    
    filtered = df.copy()
    if filter_opt == "Passed":
        filtered = filtered[filtered["passed"] == 1]
    elif filter_opt == "Failed":
        filtered = filtered[filtered["passed"] == 0]
    
    display = filtered[['test_id', 'passed', 'score', 'latency_ms']].copy()
    display['Status'] = display['passed'].apply(lambda x: '✓' if x else '✗')
    display['Latency'] = display['latency_ms'].apply(lambda x: f"{x/1000:.2f}s" if pd.notna(x) else "—")
    display['Score'] = display['score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    
    st.dataframe(
        display[['test_id', 'Status', 'Score', 'Latency']].rename(columns={'test_id': 'Test ID'}),
        use_container_width=True,
        height=300,
        hide_index=True
    )
