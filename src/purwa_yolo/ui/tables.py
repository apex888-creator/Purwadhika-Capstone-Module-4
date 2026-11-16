"""Simple HTML tables, so the app does not need to import pandas.

(pandas ships compiled DLLs; on some Windows laptops Application Control blocks
a freshly installed pandas, and this app only needed it to draw two tables.)
"""
from html import escape

import streamlit as st

CELL_STYLE = "padding:6px 10px;border-bottom:1px solid rgba(128,128,128,0.35);"


def html_table(rows, highlight_last=False):
    """Build a table from a list of dicts that all share the same keys.
    Every value is HTML-escaped. highlight_last makes the last row bold (TOTAL)."""
    if not rows:
        return ""
    headers = list(rows[0])
    head = "".join(
        f'<th style="{CELL_STYLE}text-align:left">{escape(str(h))}</th>'
        for h in headers)
    body = []
    for i, row in enumerate(rows):
        bold = "font-weight:700;" if highlight_last and i == len(rows) - 1 else ""
        cells = "".join(
            f'<td style="{CELL_STYLE}{bold}">{escape(str(row.get(h, "")))}</td>'
            for h in headers)
        body.append(f"<tr>{cells}</tr>")
    return ('<div style="overflow-x:auto">'
            '<table style="width:100%;border-collapse:collapse">'
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
            "</table></div>")


def show_table(rows, highlight_last=False):
    st.html(html_table(rows, highlight_last))
