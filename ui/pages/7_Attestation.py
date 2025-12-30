"""
Meridian UI - Attestation Management Page
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Attestation - Meridian", page_icon="M", layout="wide")

st.title("Attestation Management")

st.markdown("""
**Tamper-Evident Golden Runs** provide cryptographic verification of evaluation results.

This page allows you to:
- **Verify** - Check the integrity of an attested run
- **Export** - Create a portable zip bundle
- **Import** - Load an external bundle and verify it
- **Replay** - Re-run an evaluation with the same config
""")

# Get attestation manager
try:
    from meridian.storage.attestation import get_attestation_manager
    attester = get_attestation_manager()
    runs_dir = attester.base_dir
except Exception as e:
    st.error(f"Failed to load attestation manager: {e}")
    st.stop()

# List available attested runs
st.markdown("---")
st.subheader("Available Attested Runs")

attested_runs = []
if runs_dir.exists():
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if run_dir.is_dir():
            attestation_file = run_dir / "attestation.json"
            if attestation_file.exists():
                attested_runs.append(run_dir.name)

if not attested_runs:
    st.info("No attested runs found. Run an evaluation with 'Enable Attestation' checked.")
else:
    st.caption(f"Found {len(attested_runs)} attested runs")

# Tabs for different operations
tab1, tab2, tab3, tab4 = st.tabs(["Verify", "Export", "Import", "Replay"])

# ============================================================================
# VERIFY TAB
# ============================================================================
with tab1:
    st.subheader("Verify Attestation Integrity")
    
    if not attested_runs:
        st.warning("No attested runs available to verify.")
    else:
        selected_run = st.selectbox("Select Run to Verify", attested_runs, key="verify_run")
        
        if st.button("Verify Integrity", type="primary", key="verify_btn"):
            with st.spinner("Verifying..."):
                valid, issues = attester.verify(selected_run)
                attestation = attester.load_attestation(selected_run)
            
            if valid:
                st.success("ATTESTATION VALID - No tampering detected")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Hashes:**")
                    st.code(f"""Manifest:  {attestation.manifest_hash[:24]}...
Config:    {attestation.config_hash[:24]}...
Suite:     {attestation.suite_hash[:24]}...
Responses: {attestation.responses_hash[:24]}...""")
                
                with col2:
                    st.markdown("**Environment (Original):**")
                    env = attestation.environment
                    git_info = f"Git: {env.git_commit}" + (" (dirty)" if env.git_dirty else "") if env.git_commit else "Git: N/A"
                    st.code(f"""Created: {attestation.created_at[:19]}
Python: {env.python_version}
OS: {env.os_name}
{git_info}""")
            else:
                st.error("ATTESTATION INVALID - Tampering detected!")
                for issue in issues:
                    st.error(f"- {issue}")

# ============================================================================
# EXPORT TAB
# ============================================================================
with tab2:
    st.subheader("Export Portable Bundle")
    
    if not attested_runs:
        st.warning("No attested runs available to export.")
    else:
        selected_run = st.selectbox("Select Run to Export", attested_runs, key="export_run")
        
        if st.button("Export as ZIP", type="primary", key="export_btn"):
            import shutil
            
            with st.spinner("Exporting..."):
                # Verify first
                valid, issues = attester.verify(selected_run)
                
                if not valid:
                    st.error("Cannot export - attestation invalid")
                    for issue in issues:
                        st.error(f"- {issue}")
                else:
                    run_dir = attester.base_dir / selected_run
                    output_path = Path(f"{selected_run}.zip")
                    
                    shutil.make_archive(str(output_path.with_suffix('')), 'zip', run_dir.parent, selected_run)
                    
                    size_mb = output_path.stat().st_size / (1024 * 1024)
                    st.success(f"Exported: {output_path}")
                    st.info(f"Size: {size_mb:.2f} MB")
                    
                    # Provide download
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="Download ZIP",
                            data=f.read(),
                            file_name=output_path.name,
                            mime="application/zip"
                        )

# ============================================================================
# IMPORT TAB
# ============================================================================
with tab3:
    st.subheader("Import Bundle")
    
    uploaded_file = st.file_uploader("Upload ZIP Bundle", type=['zip'])
    
    if uploaded_file:
        if st.button("Import and Verify", type="primary", key="import_btn"):
            import zipfile
            import shutil
            import tempfile
            
            with st.spinner("Importing..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = Path(tmp.name)
                
                try:
                    # Extract and get run_id
                    with zipfile.ZipFile(tmp_path, 'r') as zf:
                        names = zf.namelist()
                        if not names:
                            st.error("Empty bundle")
                        else:
                            run_id = names[0].split('/')[0]
                            target_dir = attester.base_dir / run_id
                            
                            if target_dir.exists():
                                shutil.rmtree(target_dir)
                            
                            zf.extractall(attester.base_dir)
                            
                            # Verify
                            valid, issues = attester.verify(run_id)
                            
                            if valid:
                                attestation = attester.load_attestation(run_id)
                                st.success("Import successful - attestation valid")
                                st.info(f"Run ID: {run_id}")
                                st.info(f"Original: Python {attestation.environment.python_version} on {attestation.environment.os_name}")
                            else:
                                st.error("Import failed - attestation invalid!")
                                for issue in issues:
                                    st.error(f"- {issue}")
                                shutil.rmtree(target_dir)
                finally:
                    tmp_path.unlink()

# ============================================================================
# REPLAY TAB
# ============================================================================
with tab4:
    st.subheader("Replay Attested Run")
    
    st.markdown("""
**Replay modes:**
- **Drift** - For API models (allows statistical variance)
- **Strict** - For local models (expects exact match)
""")
    
    if not attested_runs:
        st.warning("No attested runs available to replay.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            selected_run = st.selectbox("Select Run to Replay", attested_runs, key="replay_run")
        
        with col2:
            replay_mode = st.selectbox("Replay Mode", ["drift", "strict"], key="replay_mode")
        
        # Show original config
        import json
        config_path = attester.base_dir / selected_run / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            st.markdown("**Original Configuration:**")
            st.json(config_data)
        
        st.warning("Replay will execute a new evaluation with the same configuration. This may incur API costs.")
        
        if st.button("Execute Replay", type="primary", key="replay_btn"):
            st.info("For replay, use the CLI command:")
            st.code(f"python -m meridian.cli replay --id {selected_run} --mode {replay_mode}")
            st.caption("Full replay in UI coming in a future update.")

# Footer
st.markdown("---")
st.caption("Tamper-Evident Attestation uses SHA256 hashing to detect any modifications to evaluation artifacts.")
