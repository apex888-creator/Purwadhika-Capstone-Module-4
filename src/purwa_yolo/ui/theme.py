"""Visual theme: road-signage look (asphalt + lane-marking amber).

One place for all CSS and small HTML helpers, so the other UI modules stay clean.
"""
from html import escape

import streamlit as st

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@600;700&display=swap');

:root {
  --amber: #FFB400;
  --amber-soft: rgba(255,180,0,.14);
  --asphalt: #14171C;
  --line: rgba(128,128,128,.28);
  --ok: #2ECC71;
  --warn: #FFB400;
  --bad: #FF5C5C;
}

.stApp { font-family: 'Barlow', system-ui, sans-serif; }
h1, h2, h3, .step .t, .gantry h1 {
  font-family: 'Barlow Condensed', 'Barlow', sans-serif !important;
  letter-spacing: .2px;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1080px; }

/* ---------- hero: an ERP gantry sign ---------- */
.gantry {
  background: var(--asphalt);
  border: 2px solid var(--amber);
  border-radius: 16px;
  padding: 22px 28px;
  margin-bottom: 1.4rem;
  box-shadow: 0 0 0 5px rgba(255,180,0,.10), 0 14px 34px rgba(0,0,0,.35);
  display: flex; gap: 22px; align-items: center; flex-wrap: wrap;
}
.gantry .lamp {
  width: 16px; height: 16px; border-radius: 50%;
  background: var(--amber); box-shadow: 0 0 14px 4px rgba(255,180,0,.65);
  animation: blink 2.4s ease-in-out infinite; flex: none;
}
@keyframes blink { 0%,100% {opacity:1} 50% {opacity:.35} }
@media (prefers-reduced-motion: reduce) { .gantry .lamp { animation: none; } }
.gantry h1 { margin: 0; padding: 0; font-size: 2.3rem; color: #fff; line-height: 1.05; }
.gantry p { margin: .35rem 0 0; color: rgba(255,255,255,.72); font-size: 1rem; }
.gantry .flow { margin-left: auto; display: flex; gap: 8px; flex-wrap: wrap; }
.gantry .flow span {
  background: var(--amber); color: #1a1a1a; font-weight: 600;
  padding: 5px 12px; border-radius: 6px; font-size: .85rem;
}

/* ---------- section headers ---------- */
.step {
  display: flex; align-items: center; gap: 14px;
  margin: 2.2rem 0 1rem; padding-top: 1.4rem;
  border-top: 3px dashed rgba(255,180,0,.45);   /* lane marking */
}
.step .num {
  width: 40px; height: 40px; border-radius: 8px; flex: none;
  background: var(--amber); color: #1a1a1a; font-weight: 700; font-size: 1.3rem;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Barlow Condensed', sans-serif;
}
.step .t { font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
.step .d { font-size: .9rem; opacity: .65; margin-top: 2px; }

/* ---------- chips ---------- */
.chip {
  display: inline-block; padding: 4px 12px; margin: 0 6px 6px 0;
  border-radius: 999px; font-size: .82rem; font-weight: 500;
  background: rgba(128,128,128,.14); border: 1px solid var(--line);
}
.chip.ok   { background: rgba(46,204,113,.14); border-color: rgba(46,204,113,.5); }
.chip.warn { background: var(--amber-soft);    border-color: rgba(255,180,0,.55); }
.chip.bad  { background: rgba(255,92,92,.14);  border-color: rgba(255,92,92,.55); }

/* ---------- metrics ---------- */
[data-testid="stMetric"] {
  background: rgba(128,128,128,.08);
  border: 1px solid var(--line);
  border-left: 4px solid var(--amber);
  border-radius: 12px; padding: 14px 18px;
}
[data-testid="stMetricValue"] { font-family: 'Barlow Condensed', sans-serif; font-size: 2.1rem; }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px; font-weight: 600; padding: .55rem 1.4rem;
  transition: transform .12s ease, box-shadow .12s ease;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"],
.stFormSubmitButton > button {
  background: var(--amber); color: #1a1a1a; border: none;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
  box-shadow: 0 6px 18px rgba(255,180,0,.35); color: #000;
}

/* ---------- upload, media, alerts ---------- */
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed rgba(255,180,0,.6); border-radius: 14px;
  background: var(--amber-soft);
}
[data-testid="stImage"] img, [data-testid="stVideo"] video, video { border-radius: 12px; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] [data-testid="stExpander"] {
  border: 1px solid var(--line); border-radius: 12px; margin-bottom: .6rem;
}
.brand { display:flex; align-items:center; gap:10px; margin-bottom:.6rem; }
.brand .dot { width:12px; height:12px; border-radius:50%; background:var(--amber);
              box-shadow:0 0 10px 2px rgba(255,180,0,.6); }
.brand b { font-family:'Barlow Condensed',sans-serif; font-size:1.4rem; }

/* ---------- tables ---------- */
.erp-table { width:100%; border-collapse:separate; border-spacing:0;
             border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.erp-table th { text-align:left; padding:10px 14px; font-weight:600;
                background: var(--amber-soft); border-bottom:1px solid var(--line); }
.erp-table td { padding:9px 14px; border-bottom:1px solid var(--line); }
.erp-table tbody tr:nth-child(even) { background: rgba(128,128,128,.06); }
.erp-table tbody tr:hover { background: rgba(255,180,0,.08); }
.erp-table tbody tr:last-child td { border-bottom: none; }
.erp-table tr.total td { font-weight: 700; background: var(--amber-soft); }
</style>"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle, flow=()):
    chips = "".join(f"<span>{escape(f)}</span>" for f in flow)
    st.markdown(
        f'<div class="gantry"><div class="lamp"></div>'
        f"<div><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>"
        f'<div class="flow">{chips}</div></div>',
        unsafe_allow_html=True)


def step_header(number, title, desc=""):
    d = f'<div class="d">{escape(desc)}</div>' if desc else ""
    st.markdown(
        f'<div class="step"><div class="num">{number}</div>'
        f'<div><div class="t">{escape(title)}</div>{d}</div></div>',
        unsafe_allow_html=True)


def chip(text, kind=""):
    """Return the HTML of one chip (kind: '', 'ok', 'warn' or 'bad')."""
    return f'<span class="chip {kind}">{escape(text)}</span>'


def chips(*items):
    """Render chips given as (text, kind) pairs."""
    st.markdown("".join(chip(t, k) for t, k in items), unsafe_allow_html=True)


def status(text, kind="ok"):
    chips((f"● {text}", kind))


def brand(text):
    st.markdown(f'<div class="brand"><span class="dot"></span><b>{escape(text)}</b></div>',
                unsafe_allow_html=True)
