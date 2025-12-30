"""
Meridian UI - Certification Page

Certify providers and evaluation runs to generate verifiable badges.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from meridian.config import RESULTS_DIR

st.set_page_config(page_title="Certification", page_icon="M", layout="wide")

st.title("Certification")

st.markdown("""
Generate verifiable badges for your model adapters and evaluation results.
""")

# ============================================================================
# API KEY SECTION (subtle, collapsible)
# ============================================================================
with st.expander("API Configuration", expanded=False):
    st.caption("Enter your API keys if not already set as environment variables.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        deepseek_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            placeholder="sk-...",
            key="deepseek_key"
        )
        if deepseek_key:
            import os
            os.environ["DEEPSEEK_API_KEY"] = deepseek_key
    
    with col2:
        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-proj-...",
            key="openai_key"
        )
        if openai_key:
            import os
            os.environ["OPENAI_API_KEY"] = openai_key
    
    with col3:
        mistral_key = st.text_input(
            "Mistral API Key",
            type="password",
            placeholder="...",
            key="mistral_key"
        )
        if mistral_key:
            import os
            os.environ["MISTRAL_API_KEY"] = mistral_key

st.markdown("---")

# Tabs for the two certification types
tab1, tab2 = st.tabs(["Provider Certification", "Suite Certification"])

# ============================================================================
# PROVIDER CERTIFICATION TAB
# ============================================================================
with tab1:
    st.subheader("Provider Certification")
    
    st.markdown("""
    Verify that a model adapter works correctly by running 14 standardized tests.
    A score of 80% or higher means the adapter is production-ready.
    
    **Tests include:** connectivity, basic math, temperature 0, determinism, 
    latency, max tokens, error handling, JSON mode, context window, and more.
    """)
    
    # Only the 3 models we actually use
    available_models = ["deepseek_chat", "openai_gpt4", "mistral_medium"]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help="Choose the model adapter to test"
        )
    
    with col2:
        save_report = st.checkbox("Save report", value=True)
    
    if st.button("Run Certification", type="primary", key="certify_provider"):
        from meridian.certification import certify_provider, save_certification, generate_badge_markdown
        
        with st.spinner(f"Testing '{selected_model}'... (14 tests, ~1-2 minutes)"):
            try:
                cert = certify_provider(selected_model)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Score", f"{cert.score}%")
                with col2:
                    passed = sum(1 for t in cert.tests if t.passed)
                    st.metric("Tests Passed", f"{passed}/14")
                with col3:
                    status = "CERTIFIED" if cert.overall_passed else "FAILED"
                    st.metric("Status", status)
                with col4:
                    st.metric("Badge Hash", cert.badge_hash[:8])
                
                # Detailed test results
                st.markdown("**Test Details**")
                
                results_data = []
                for test in cert.tests:
                    results_data.append({
                        "Test": test.test_name,
                        "Status": "PASS" if test.passed else "FAIL",
                        "Message": test.message[:60] if test.message else ""
                    })
                
                st.dataframe(results_data, use_container_width=True)
                
                # Badge
                st.markdown("**Your Badge**")
                badge_md = generate_badge_markdown(cert)
                st.code(badge_md, language="markdown")
                st.markdown(badge_md)
                
                # Save if requested
                if save_report:
                    report_path = save_certification(cert)
                    st.success(f"Saved: {report_path}")
                    
            except Exception as e:
                st.error(f"Certification failed: {e}")
                st.caption("Make sure your API key is configured above.")

# ============================================================================
# SUITE CERTIFICATION TAB
# ============================================================================
with tab2:
    st.subheader("Suite Certification")
    
    st.markdown("""
    Generate a verification badge for your evaluation results. This proves your 
    model achieved a specific accuracy on a test suite, with cryptographic verification.
    
    **Use this to:** Share verified results in your README, prove accuracy to clients, 
    or document model performance for audits.
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
        No attested runs found.
        
        To create one:
        1. Go to "Run Suite" page
        2. Check "Enable Attestation"
        3. Run any suite
        
        Or use CLI: `python -m meridian.cli run --suite <name> --model <model> --attest`
        """)
    else:
        st.caption(f"Found {len(attested_runs)} attested runs")
        
        # Run selection
        selected_run = st.selectbox(
            "Select Run",
            attested_runs,
            help="Choose the evaluation run to certify"
        )
        
        save_badge = st.checkbox("Save badge", value=True, key="save_suite_badge")
        
        if st.button("Generate Badge", type="primary", key="certify_suite"):
            from meridian.certification import (
                certify_suite_run,
                save_suite_certification,
                generate_suite_badge_markdown
            )
            
            with st.spinner("Generating..."):
                try:
                    cert = certify_suite_run(selected_run)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Suite", cert.suite_name)
                    with col2:
                        st.metric("Model", cert.model_id)
                    with col3:
                        st.metric("Accuracy", f"{cert.accuracy:.1f}%")
                    with col4:
                        status = "Verified" if cert.verified else "Unverified"
                        st.metric("Status", status)
                    
                    # Details
                    st.markdown(f"**Tests:** {cert.passed_tests}/{cert.total_tests} passed")
                    st.markdown(f"**Attestation Hash:** `{cert.attestation_hash}`")
                    st.markdown(f"**Badge Hash:** `{cert.badge_hash}`")
                    
                    # Badge
                    st.markdown("**Your Badge**")
                    badge_md = generate_suite_badge_markdown(cert)
                    st.code(badge_md, language="markdown")
                    st.markdown(badge_md)
                    
                    st.caption("Copy the markdown above to embed in your README.")
                    
                    # Save if requested
                    if save_badge:
                        report_path = save_suite_certification(cert)
                        st.success(f"Saved: {report_path}")
                        
                        svg_path = report_path.with_name(report_path.stem + "_badge.svg")
                        
                        # Download button
                        with open(svg_path, 'r', encoding='utf-8') as f:
                            svg_content = f.read()
                        
                        st.download_button(
                            label="Download SVG",
                            data=svg_content,
                            file_name=svg_path.name,
                            mime="image/svg+xml"
                        )
                        
                except Exception as e:
                    st.error(f"Failed: {e}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("Documentation: docs/CERTIFICATION.md")
