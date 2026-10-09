"""
Dashboard Styling and Design System
====================================
Institutional color palette, typography tokens, custom CSS, and Plotly templates.
Palette: Navy (#0F172A), Slate (#334155, #64748B), Royal Blue (#2563EB), Accent Teal (#0D9488).
"""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

# Custom CSS for Streamlit
CUSTOM_CSS = """
<style>
/* Import Inter Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Main Container Adjustments */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1300px;
}

/* Metric Card */
.fin-metric-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 1px 3px 0 rgba(15, 23, 42, 0.05);
    margin-bottom: 0.75rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.fin-metric-card:hover {
    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.08);
}
.fin-metric-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748B;
    margin-bottom: 0.35rem;
}
.fin-metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
    font-family: 'JetBrains Mono', monospace;
}
.fin-metric-subtext {
    font-size: 0.8rem;
    color: #475569;
    margin-top: 0.35rem;
}
.fin-metric-delta-pos {
    color: #16A34A;
    font-weight: 600;
}
.fin-metric-delta-neg {
    color: #DC2626;
    font-weight: 600;
}

/* Section Header */
.fin-section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #0F172A;
    margin-top: 1.5rem;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.fin-section-subtitle {
    font-size: 0.875rem;
    color: #64748B;
    margin-bottom: 1.25rem;
}

/* Notice / Disclaimer Banner */
.fin-disclaimer-box {
    background: #F8FAFC;
    border-left: 4px solid #0284C7;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    font-size: 0.825rem;
    color: #334155;
    margin-bottom: 1.25rem;
    line-height: 1.5;
}
.fin-warning-box {
    background: #FFFBEB;
    border-left: 4px solid #D97706;
    border-radius: 4px;
    padding: 0.75rem 1rem;
    font-size: 0.825rem;
    color: #92400E;
    margin-bottom: 1.25rem;
}

/* Custom Badges */
.badge-pill {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    font-size: 0.75rem;
    font-weight: 600;
    border-radius: 9999px;
    margin-left: 0.35rem;
}
.badge-blue { background: #DBEAFE; color: #1D4ED8; }
.badge-green { background: #DCFCE7; color: #15803D; }
.badge-amber { background: #FEF3C7; color: #B45309; }
.badge-slate { background: #F1F5F9; color: #475569; }

/* Table styling tweaks */
div[data-testid="stDataFrame"] {
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    overflow: hidden;
}
</style>
"""


def apply_theme():
    """Inject custom institutional CSS stylesheet into the active Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_metric_card(
    title: str,
    value: str,
    subtext: str = "",
    delta: str = "",
    delta_positive: bool = True,
    badge: str = "",
) -> str:
    """Returns pure HTML string for a polished financial metric card."""
    delta_html = ""
    if delta:
        cls = "fin-metric-delta-pos" if delta_positive else "fin-metric-delta-neg"
        delta_html = f"<span class='{cls}'>{delta}</span> · "

    badge_html = f"<span class='badge-pill badge-blue'>{badge}</span>" if badge else ""

    return f"""
    <div class="fin-metric-card">
        <div class="fin-metric-title">{title} {badge_html}</div>
        <div class="fin-metric-value">{value}</div>
        <div class="fin-metric-subtext">{delta_html}{subtext}</div>
    </div>
    """


def render_section_header(title: str, subtitle: str = ""):
    """Renders a styled section header."""
    st.markdown(f'<div class="fin-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="fin-section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def render_disclaimer(text: str, is_warning: bool = False):
    """Renders an institutional compliance or methodological caveat banner."""
    cls = "fin-warning-box" if is_warning else "fin-disclaimer-box"
    icon = "⚠️" if is_warning else "ℹ️"
    st.markdown(
        f'<div class="{cls}"><strong>{icon} Compliance & Scope:</strong> {text}</div>',
        unsafe_allow_html=True,
    )


def get_plotly_layout(
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 440,
    legend_top: bool = False,
) -> dict:
    """
    Returns an institutional-grade clean layout specification for Plotly figures.
    """
    layout = dict(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Inter, sans-serif", size=15, color="#0F172A"),
            x=0.01,
            xanchor="left",
        ),
        font=dict(family="Inter, sans-serif", color="#475569", size=12),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FAFAFC",
        height=height,
        margin=dict(l=55, r=25, t=55 if title else 25, b=50),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(size=12, color="#475569")),
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            showline=True,
            linecolor="#CBD5E1",
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(size=12, color="#475569")),
            gridcolor="#F1F5F9",
            zerolinecolor="#E2E8F0",
            showline=True,
            linecolor="#CBD5E1",
            tickfont=dict(size=11),
        ),
        legend=dict(
            orientation="h" if legend_top else "h",
            yanchor="bottom" if legend_top else "bottom",
            y=1.02 if legend_top else -0.22,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font=dict(family="JetBrains Mono, monospace", size=12, color="#FFFFFF"),
            bordercolor="#0F172A",
        ),
    )
    return layout
