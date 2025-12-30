"""
Meridian UI - Create Suite Page

Upload and validate custom evaluation suites.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.set_page_config(page_title="Create Suite", page_icon="M", layout="wide")

st.title("Create Suite")

st.markdown("""
Upload your own prompts to create a custom evaluation suite. 
Your suite will be validated for common issues before saving.
""")

# ============================================================================
# UPLOAD SECTION
# ============================================================================

st.subheader("Upload Test Cases")

upload_format = st.radio(
    "Format",
    ["JSONL", "CSV"],
    horizontal=True,
    help="JSONL: one JSON object per line. CSV: with headers."
)

st.markdown("""
**Required fields:**
- `input` or `prompt`: The prompt to send to the model
- `expected` or `answer`: The expected response (optional for llm_judge)

**Optional fields:**
- `id`: Unique identifier (auto-generated if missing)
- `scorer`: `exact`, `contains`, `regex`, or `llm_judge` (default: exact)
- `rubric`: Required if scorer is `llm_judge`
- `tags`: Comma-separated tags for filtering
""")

# Example templates
with st.expander("See example format"):
    if upload_format == "JSONL":
        st.code('''{"id": "math_1", "input": "What is 2+2?", "expected": "4", "scorer": "exact"}
{"id": "translate_1", "input": "Translate 'hello' to Spanish", "expected": "hola", "scorer": "contains"}
{"id": "essay_1", "input": "Write about AI ethics", "scorer": "llm_judge", "rubric": "Score 1-5 on clarity and depth"}''', language="json")
    else:
        st.code('''id,input,expected,scorer
math_1,"What is 2+2?",4,exact
translate_1,"Translate 'hello' to Spanish",hola,contains''', language="csv")

# File upload
uploaded_file = st.file_uploader(
    f"Upload {upload_format} file",
    type=["jsonl", "json", "csv"] if upload_format == "JSONL" else ["csv"]
)

# Manual input option
st.markdown("**Or paste content directly:**")
manual_content = st.text_area(
    "Paste content",
    height=150,
    placeholder=f"Paste your {upload_format} content here..."
)

# Suite metadata
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    suite_name = st.text_input(
        "Suite Name",
        placeholder="my_evaluation_suite",
        help="Unique name for your suite"
    )

with col2:
    suite_description = st.text_input(
        "Description (optional)",
        placeholder="Tests for customer support responses"
    )

# ============================================================================
# VALIDATION AND PREVIEW
# ============================================================================

if st.button("Validate and Preview", type="primary"):
    content = None
    
    if uploaded_file:
        content = uploaded_file.read().decode("utf-8")
    elif manual_content.strip():
        content = manual_content.strip()
    
    if not content:
        st.error("Please upload a file or paste content.")
    elif not suite_name:
        st.error("Please enter a suite name.")
    else:
        from meridian.suites.custom import EvaluationPack, get_custom_suite_manager
        
        try:
            # Parse based on format
            if upload_format == "JSONL":
                pack = EvaluationPack.from_jsonl(suite_name, content, suite_description)
            else:
                pack = EvaluationPack.from_csv(suite_name, content, suite_description)
            
            # Validate
            issues = pack.validate()
            
            # Display results
            st.markdown("### Validation Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Test Cases", len(pack.tests))
            with col2:
                st.metric("Issues Found", len(issues))
            with col3:
                status = "Ready" if not issues else "Has Warnings"
                st.metric("Status", status)
            
            # Show issues
            if issues:
                st.warning("**Validation Issues:**")
                for issue in issues:
                    if "LEAK WARNING" in issue:
                        st.error(f"- {issue}")
                    else:
                        st.warning(f"- {issue}")
            else:
                st.success("No issues found.")
            
            # Preview tests
            st.markdown("### Preview")
            
            preview_data = []
            for test in pack.tests[:10]:  # Show first 10
                preview_data.append({
                    "ID": test.id,
                    "Input": test.input[:50] + "..." if len(test.input) > 50 else test.input,
                    "Expected": (test.expected[:30] + "...") if test.expected and len(test.expected) > 30 else test.expected,
                    "Scorer": test.scorer
                })
            
            st.dataframe(preview_data, use_container_width=True)
            
            if len(pack.tests) > 10:
                st.caption(f"Showing 10 of {len(pack.tests)} tests")
            
            # Holdout split info
            dev, eval_set = pack.get_holdout_split()
            st.info(f"**Holdout split:** {len(dev)} dev / {len(eval_set)} eval (seed=42)")
            
            # Save button
            st.markdown("---")
            
            # Check for existing suite with same name
            manager = get_custom_suite_manager()
            existing = manager.get_by_name(suite_name)
            
            if existing:
                st.warning(f"A suite named '{suite_name}' already exists. Saving will overwrite it.")
            
            if st.button("Save Suite", type="primary", key="save_suite"):
                suite_id = manager.save(pack)
                st.success(f"Suite saved. ID: `{suite_id}`")
                st.info("Go to 'Run Suite' to execute your evaluation.")
                
        except ValueError as e:
            st.error(f"Parse error: {e}")
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================================
# EXISTING SUITES
# ============================================================================

st.markdown("---")
st.subheader("Your Suites")

from meridian.suites.custom import get_custom_suite_manager

manager = get_custom_suite_manager()
suites = manager.list_all()

if not suites:
    st.info("No custom suites yet. Upload one above.")
else:
    for suite in suites:
        with st.expander(f"{suite['name']} ({suite['test_count']} tests)"):
            st.markdown(f"**Description:** {suite['description'] or 'None'}")
            st.markdown(f"**Created:** {suite['created_at'][:10]}")
            st.markdown(f"**ID:** `{suite['id']}`")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Export JSONL", key=f"export_{suite['id']}"):
                    content = manager.export_jsonl(suite['id'])
                    st.download_button(
                        "Download",
                        content,
                        file_name=f"{suite['name']}.jsonl",
                        mime="application/jsonl",
                        key=f"dl_{suite['id']}"
                    )
            
            with col2:
                if st.button("Delete", key=f"delete_{suite['id']}", type="secondary"):
                    if manager.delete(suite['id']):
                        st.success("Deleted")
                        st.rerun()

# Footer
st.markdown("---")
st.caption("Suites are stored locally in SQLite. Use Export to share.")
