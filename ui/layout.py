"""
Meridian UI - Layout Components
Reusable premium components
"""

import streamlit as st

# Design tokens
DARK = "#111827"
GRAY = "#6B7280"
LIGHT = "#9CA3AF"
BORDER = "#E5E7EB"
BG = "#F8FAFC"
GOOD = "#065F46"
BAD = "#991B1B"


def inject_global_style():
    """Inject premium CSS into page"""
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
      
      html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      }
      
      .block-container { padding-top: 2.0rem; max-width: 1200px; }
      h1, h2, h3 { letter-spacing: -0.02em; color: #111827; }
      
      .ce-subtitle { color: #6B7280; font-size: 14px; margin-top: -8px; margin-bottom: 20px; }
      
      .ce-row { display: flex; gap: 14px; align-items: stretch; flex-wrap: wrap; }
      
      .ce-card {
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 16px 18px;
        background: white;
        flex: 1;
        min-width: 180px;
      }
      
      .ce-kpi {
        font-size: 36px;
        font-weight: 700;
        color: #111827;
        line-height: 1.0;
      }
      .ce-kpi-label { 
        color: #6B7280; 
        font-size: 11px; 
        margin-top: 6px; 
        text-transform: uppercase; 
        letter-spacing: 0.08em;
        font-weight: 500;
      }
      .ce-muted { color: #6B7280; font-size: 12px; margin-top: 4px; }
      
      .ce-pill {
        display: inline-flex; 
        gap: 6px; 
        align-items: center;
        border: 1px solid #E5E7EB; 
        border-radius: 999px;
        padding: 5px 12px; 
        background: #F8FAFC;
        color: #111827; 
        font-size: 12px;
        font-weight: 500;
      }
      
      .ce-divider { height: 1px; background: #E5E7EB; margin: 24px 0; }
      
      .ce-callout {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 14px 16px;
        background: #F8FAFC;
        color: #111827;
        font-size: 13px;
        line-height: 1.5;
      }
      
      .ce-section-title {
        font-size: 11px;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 28px 0 14px 0;
      }
      
      .ce-good { color: #065F46; }
      .ce-bad { color: #991B1B; }
      
      .ce-failure-card {
        border: 1px solid #E5E7EB;
        border-left: 3px solid #991B1B;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        background: white;
        margin-bottom: 10px;
      }
      .ce-failure-title {
        font-size: 13px;
        font-weight: 600;
        color: #111827;
        font-family: 'SF Mono', Monaco, monospace;
      }
      .ce-failure-reason {
        font-size: 12px;
        color: #6B7280;
        margin-top: 4px;
      }
    </style>
    """, unsafe_allow_html=True)


def kpi_card(value: str, label: str, hint: str = None, tone: str = "neutral") -> str:
    """
    Generate HTML for a KPI card
    tone: "neutral" | "good" | "bad"
    """
    cls = "ce-kpi"
    if tone == "good":
        cls += " ce-good"
    elif tone == "bad":
        cls += " ce-bad"
    
    hint_html = f'<div class="ce-muted">{hint}</div>' if hint else ""
    
    return f"""
    <div class="ce-card">
      <div class="{cls}">{value}</div>
      <div class="ce-kpi-label">{label}</div>
      {hint_html}
    </div>
    """


def pills_bar(items: list) -> str:
    """Generate context pills bar"""
    pills = " ".join([f'<span class="ce-pill">{x}</span>' for x in items])
    return f'<div style="display:flex; flex-wrap:wrap; gap:8px; margin: 12px 0;">{pills}</div>'


def section_title(text: str) -> str:
    """Section title HTML"""
    return f'<div class="ce-section-title">{text}</div>'


def divider() -> str:
    """Divider HTML"""
    return '<div class="ce-divider"></div>'


def callout(text: str) -> str:
    """Callout box HTML"""
    return f'<div class="ce-callout">{text}</div>'


def failure_card(test_id: str, reason: str = None) -> str:
    """Failure card HTML"""
    reason_html = f'<div class="ce-failure-reason">{reason}</div>' if reason else ""
    return f"""
    <div class="ce-failure-card">
      <div class="ce-failure-title">{test_id}</div>
      {reason_html}
    </div>
    """
