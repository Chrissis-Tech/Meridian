"""
Meridian UI - Compare Page (Framework Top)
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ui.layout import inject_global_style, kpi_card, pills_bar, section_title, divider, callout
from ui.vega_theme import dumbbell, latency_boxplot

st.set_page_config(page_title="Compare - Meridian", page_icon="M", layout="wide")
inject_global_style()

st.markdown("# Compare Runs")
st.markdown('<div class="ce-subtitle">A/B comparison with statistical context and regression detection.</div>', unsafe_allow_html=True)

try:
    from core.storage.db import get_db
    from core.runner import SuiteRunner
    
    db = get_db()
    runs = db.get_runs(limit=50)
except Exception as e:
    st.error(f"Error: {e}")
    runs = []

if len(runs) < 2:
    st.markdown(callout("Need at least 2 runs to compare. Go to <b>Run Suite</b> first."), unsafe_allow_html=True)
    st.stop()

run_options = {r["run_id"]: f"{r['suite_name']} · {r['model_id']} · {r['started_at'][:10]}" for r in runs}
run_suites = {r["run_id"]: r["suite_name"] for r in runs}
run_models = {r["run_id"]: r["model_id"] for r in runs}

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Baseline (A)**")
    run_a_id = st.selectbox("A", list(run_options.keys()), format_func=lambda x: run_options[x], label_visibility="collapsed")

with col2:
    st.markdown("**Comparison (B)**")
    run_b_id = st.selectbox("B", list(run_options.keys()), format_func=lambda x: run_options[x], index=min(1, len(runs)-1), label_visibility="collapsed")

if st.button("Compare", type="primary"):
    runner = SuiteRunner()
    comparison = runner.compare_runs(run_a_id, run_b_id)
    
    run_a = db.get_run(run_a_id)
    run_b = db.get_run(run_b_id)
    
    acc_a = run_a['accuracy'] * 100 if run_a['accuracy'] else 0
    acc_b = run_b['accuracy'] * 100 if run_b['accuracy'] else 0
    delta = acc_b - acc_a
    
    results_a_list = db.get_results(run_a_id)
    results_b_list = db.get_results(run_b_id)
    
    lat_a = [r['latency_ms']/1000 for r in results_a_list if r['latency_ms']]
    lat_b = [r['latency_ms']/1000 for r in results_b_list if r['latency_ms']]
    
    model_a = run_models.get(run_a_id, "A")
    model_b = run_models.get(run_b_id, "B")
    
    same_suite = run_suites.get(run_a_id) == run_suites.get(run_b_id)
    n_common = len(set(r["test_id"] for r in results_a_list) & set(r["test_id"] for r in results_b_list))
    
    st.markdown(divider(), unsafe_allow_html=True)
    
    # Context pills
    st.markdown(pills_bar([
        f"Suite: {run_suites.get(run_a_id, '?')}",
        f"A: {model_a}",
        f"B: {model_b}",
        f"n={n_common} paired tests"
    ]), unsafe_allow_html=True)
    
    # KPI cards
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown(kpi_card(f"{acc_a:.0f}%", f"A: {model_a}"), unsafe_allow_html=True)
    
    with c2:
        st.markdown(kpi_card(f"{acc_b:.0f}%", f"B: {model_b}"), unsafe_allow_html=True)
    
    with c3:
        delta_str = f"+{delta:.0f}%" if delta > 0 else f"{delta:.0f}%"
        tone = "good" if delta > 0 else "bad" if delta < 0 else "neutral"
        st.markdown(kpi_card(delta_str, "Δ Accuracy", tone=tone), unsafe_allow_html=True)
    
    # Statistical note
    if same_suite and n_common >= 10:
        note = f"n={n_common} paired tests · Bootstrap B=1000"
    else:
        note = f"McNemar skipped (n={n_common} paired samples, need ≥10 for significance)"
    
    st.markdown(callout(note), unsafe_allow_html=True)
    
    st.markdown(divider(), unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown(section_title("Accuracy Comparison"), unsafe_allow_html=True)
        st.vega_lite_chart(dumbbell(acc_a, acc_b, model_a, model_b, ""), use_container_width=True)
    
    with col2:
        st.markdown(section_title("Latency Distribution"), unsafe_allow_html=True)
        if lat_a and lat_b:
            st.vega_lite_chart(latency_boxplot(lat_a, lat_b, model_a, model_b, ""), use_container_width=True)
            
            p50_a = np.percentile(lat_a, 50)
            p50_b = np.percentile(lat_b, 50)
            faster = model_a if p50_a < p50_b else model_b
            ratio = max(p50_a, p50_b) / min(p50_a, p50_b) if min(p50_a, p50_b) > 0 else 1
            st.caption(f"**{faster}** is {ratio:.1f}x faster (by p50)")
    
    # Test changes
    if same_suite:
        st.markdown(divider(), unsafe_allow_html=True)
        st.markdown(section_title("Test Changes"), unsafe_allow_html=True)
        
        results_a = {r["test_id"]: r for r in results_a_list}
        results_b = {r["test_id"]: r for r in results_b_list}
        common_ids = set(results_a.keys()) & set(results_b.keys())
        
        improved = [t for t in common_ids if not results_a[t]["passed"] and results_b[t]["passed"]]
        regressed = [t for t in common_ids if results_a[t]["passed"] and not results_b[t]["passed"]]
        unchanged = len(common_ids) - len(improved) - len(regressed)
        
        st.markdown(pills_bar([
            f"{len(improved)} improved",
            f"{len(regressed)} regressed",
            f"{unchanged} unchanged"
        ]), unsafe_allow_html=True)
        
        # Decision box
        if delta >= 5 and len(regressed) == 0:
            st.success(f"**Recommendation: Ship B** — {delta:.0f}% improvement with no regressions")
        elif delta <= -5 or len(regressed) > len(improved):
            st.error(f"**Recommendation: Stay with A** — B has {len(regressed)} regressions")
        else:
            st.info(f"**Recommendation: Review** — Consider latency and specific test failures")
        
        # Changed tests detail
        if improved or regressed:
            with st.expander(f"View {len(improved) + len(regressed)} changed tests"):
                col1, col2 = st.columns(2)
                with col1:
                    if improved:
                        st.markdown("**Improved** (A failed → B passed)")
                        for t in improved[:5]:
                            st.code(t, language=None)
                with col2:
                    if regressed:
                        st.markdown("**Regressed** (A passed → B failed)")
                        for t in regressed[:5]:
                            st.code(t, language=None)
