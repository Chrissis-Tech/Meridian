"""
Meridian UI - Certification Page

Certify providers and evaluation runs to generate verifiable badges.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from meridian.config import RESULTS_DIR

st.set_page_config(page_title="Certification", page_icon="🏆", layout="wide")

st.title("🏆 Certification")

st.markdown("""
Meridian provides two types of certification:

1. **Provider Certification** - Verify that a model adapter works correctly (14 tests)
2. **Suite Certification** - Generate a verification badge for your evaluation results
""")

# Tabs for the two certification types
tab1, tab2 = st.tabs(["🔌 Provider Certification", "📊 Suite Certification"])

# ============================================================================
# PROVIDER CERTIFICATION TAB
# ============================================================================
with tab1:
    st.subheader("Provider Certification")
    
    st.info("""
    **What is this?**
    
    Provider Certification runs 14 standardized tests against a model adapter to verify it works correctly:
    
    - **Basic tests (7):** connectivity, basic math, temperature 0, determinism, latency, max tokens, error handling
    - **Advanced tests (7):** system prompt, JSON mode, context window (2K tokens), unicode, empty prompt, long output, special characters
    
    A score of **80%+ = CERTIFIED** (production-ready).
    """)
    
    # Model selection
    from meridian.model_adapters import get_available_models
    
    available_models = get_available_models()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select Model to Certify",
            available_models,
            help="Choose the model adapter you want to test"
        )
    
    with col2:
        save_report = st.checkbox("Save report & badge", value=True)
    
    if st.button("🚀 Run Certification", type="primary", key="certify_provider"):
        from meridian.certification import certify_provider, save_certification, generate_badge_markdown
        
        with st.spinner(f"Certifying '{selected_model}'... (14 tests, may take 1-2 minutes)"):
            try:
                cert = certify_provider(selected_model)
                
                # Display results
                st.markdown("### Results")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Score", f"{cert.score}%")
                with col2:
                    passed = sum(1 for t in cert.tests if t.passed)
                    st.metric("Tests Passed", f"{passed}/14")
                with col3:
                    status = "CERTIFIED ✅" if cert.overall_passed else "FAILED ❌"
                    st.metric("Status", status)
                with col4:
                    st.metric("Badge Hash", cert.badge_hash[:8])
                
                # Detailed test results
                st.markdown("### Test Details")
                
                results_data = []
                for test in cert.tests:
                    results_data.append({
                        "Test": test.test_name,
                        "Status": "✅ PASS" if test.passed else "❌ FAIL",
                        "Message": test.message[:60] if test.message else ""
                    })
                
                st.dataframe(results_data, use_container_width=True)
                
                # Badge
                st.markdown("### Your Badge")
                badge_md = generate_badge_markdown(cert)
                st.code(badge_md, language="markdown")
                st.markdown(badge_md)
                
                # Save if requested
                if save_report:
                    report_path = save_certification(cert)
                    st.success(f"Report saved: `{report_path}`")
                    st.success(f"Badge saved: `{report_path.with_suffix('.svg')}`")
                    
            except Exception as e:
                st.error(f"Certification failed: {e}")

# ============================================================================
# SUITE CERTIFICATION TAB
# ============================================================================
with tab2:
    st.subheader("Suite Certification")
    
    st.info("""
    **What is this?**
    
    Suite Certification generates a verifiable badge for your evaluation runs. This proves:
    
    - **Accuracy** - What percentage of tests passed
    - **Verified** - Whether the attestation is valid (tamper-evident)
    - **Badge Hash** - Unique identifier for this specific result
    
    You can embed this badge in your README to prove your model achieved a certain accuracy.
    """)
    
    # Find attested runs
    attested_runs = []
    if RESULTS_DIR.exists():
        for run_dir in RESULTS_DIR.iterdir():
            if run_dir.is_dir():
                attestation_file = run_dir / "attestation.json"
                if attestation_file.exists():
                    attested_runs.append(run_dir.name)
    
    if not attested_runs:
        st.warning("""
        **No attested runs found.**
        
        To create an attested run:
        1. Go to "Run Suite" page
        2. Check "Enable Attestation"
        3. Run a suite
        
        Or use CLI: `python -m meridian.cli run --suite <name> --model <model> --attest`
        """)
    else:
        st.caption(f"Found {len(attested_runs)} attested runs")
        
        # Run selection
        selected_run = st.selectbox(
            "Select Attested Run",
            attested_runs,
            help="Choose the run you want to certify"
        )
        
        save_badge = st.checkbox("Save badge & report", value=True, key="save_suite_badge")
        
        if st.button("🏅 Generate Badge", type="primary", key="certify_suite"):
            from meridian.certification import (
                certify_suite_run,
                save_suite_certification,
                generate_suite_badge_markdown
            )
            
            with st.spinner("Generating certification..."):
                try:
                    cert = certify_suite_run(selected_run)
                    
                    # Display results
                    st.markdown("### Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Suite", cert.suite_name)
                    with col2:
                        st.metric("Model", cert.model_id)
                    with col3:
                        st.metric("Accuracy", f"{cert.accuracy:.1f}%")
                    with col4:
                        status = "✅ Verified" if cert.verified else "❓ Unverified"
                        st.metric("Attestation", status)
                    
                    # More details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Test Results:**")
                        st.write(f"Passed: {cert.passed_tests}/{cert.total_tests}")
                    
                    with col2:
                        st.markdown("**Hashes:**")
                        st.code(f"Attestation: {cert.attestation_hash}\nBadge: {cert.badge_hash}")
                    
                    # Badge
                    st.markdown("### Your Badge")
                    badge_md = generate_suite_badge_markdown(cert)
                    st.code(badge_md, language="markdown")
                    st.markdown(badge_md)
                    
                    st.caption("Copy the markdown above to embed in your README!")
                    
                    # Save if requested
                    if save_badge:
                        report_path = save_suite_certification(cert)
                        st.success(f"Report saved: `{report_path}`")
                        svg_path = report_path.with_name(report_path.stem + "_badge.svg")
                        st.success(f"Badge saved: `{svg_path}`")
                        
                        # Download button for SVG
                        with open(svg_path, 'r', encoding='utf-8') as f:
                            svg_content = f.read()
                        
                        st.download_button(
                            label="📥 Download SVG Badge",
                            data=svg_content,
                            file_name=svg_path.name,
                            mime="image/svg+xml"
                        )
                        
                except Exception as e:
                    st.error(f"Certification failed: {e}")

# ============================================================================
# HELP SECTION
# ============================================================================
st.markdown("---")
st.markdown("### 📖 Quick Reference")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Provider Certification CLI:**
    ```bash
    python -m meridian.cli certify --model deepseek_chat --save
    ```
    """)

with col2:
    st.markdown("""
    **Suite Certification CLI:**
    ```bash
    python -m meridian.cli certify-run --id <run_id> --save
    ```
    """)

st.caption("Full documentation: [docs/CERTIFICATION.md](https://github.com/Chrissis-Tech/Meridian/blob/main/docs/CERTIFICATION.md)")
